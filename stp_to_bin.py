import argparse
import pathlib
import signal
import os
from multiprocessing.pool import Pool
from itertools import repeat

import dgl
import numpy as np
import torch
from occwl.graph import face_adjacency
from occwl.io import load_step
from occwl.uvgrid import ugrid, uvgrid
from tqdm import tqdm
from torch import FloatTensor

# 面类型映射（8种类型）
FACE_TYPE_MAP = {
    "plane": 0, "cylinder": 1, "cone": 2, "sphere": 3,
    "torus": 4, "bezier": 5, "bspline": 6, "nurbs": 7
}

# 边类型映射（8种类型）
EDGE_TYPE_MAP = {
    "line": 0, "circle": 1, "ellipse": 2, "parabola": 3,
    "hyperbola": 4, "bezier": 5, "bspline": 6, "nurbs": 7
}

# 凸度映射（凹、凸、平滑）
CONVEXITY_MAP = {"concave": 0, "convex": 1, "smooth": 2}

def build_graph(solid, curv_num_u_samples=5, surf_num_u_samples=5, surf_num_v_samples=5):
    """
    构建B-rep图的DGL表示，包含面节点和边特征
    
    参数:
        solid: OCCWL实体对象
        curv_num_u_samples: 曲线采样点数（默认5）
        surf_num_u_samples: 曲面u方向采样点数（默认5）
        surf_num_v_samples: 曲面v方向采样点数（默认5）
    
    返回:
        dgl.DGLGraph: 包含所有特征的DGL图
    """
    # 构建面邻接图
    graph = face_adjacency(solid)
    
    # 准备面特征列表
    graph_face_feat = []    # UV网格几何特征 [n, n, 7]
    face_types = []         # 面类型
    face_areas = []         # 面面积
    face_loops = []         # 环数量
    face_adjs = []          # 相邻面数量
    
    # 处理每个面
    for face_idx in graph.nodes:
        face = graph.nodes[face_idx]["face"]
        
        try:
            # UV网格采样
            points = uvgrid(
                face, method="point", num_u=surf_num_u_samples, num_v=surf_num_v_samples
            )
            normals = uvgrid(
                face, method="normal", num_u=surf_num_u_samples, num_v=surf_num_v_samples
            )
            visibility_status = uvgrid(
                face, method="visibility_status", num_u=surf_num_u_samples, num_v=surf_num_v_samples
            )
            
            # 处理mask维度
            mask = np.logical_or(visibility_status == 0, visibility_status == 2)
            
            # 标准化维度
            if mask.ndim > 2:
                mask = mask.squeeze()
            if mask.ndim == 2:
                mask = mask.astype(np.float32)[..., np.newaxis]
            else:
                mask = mask.reshape(surf_num_u_samples, surf_num_v_samples, 1).astype(np.float32)
            
            # 确保所有数组维度一致
            if points.ndim != 3:
                points = points.reshape(surf_num_u_samples, surf_num_v_samples, -1)
            if normals.ndim != 3:
                normals = normals.reshape(surf_num_u_samples, surf_num_v_samples, -1)
            
            # 拼接特征
            face_feat = np.concatenate((points, normals, mask), axis=-1)
            
        except Exception as e:
            print(f"Error processing face {face_idx}: {str(e)}")
            # 创建默认特征
            face_feat = np.zeros((surf_num_u_samples, surf_num_v_samples, 7), dtype=np.float32)
        
        graph_face_feat.append(face_feat)
        
        # 属性特征提取
        try:
            surface_type = str(face.surface_type()).lower()
            face_type = FACE_TYPE_MAP.get(surface_type, 0)
        except:
            face_type = 0
        face_types.append(face_type)
        
        try:
            area = face.area()
            face_areas.append(area)
        except:
            face_areas.append(0.0)
        
        try:
            loop_count = face.number_of_loops()
            face_loops.append(loop_count)
        except:
            face_loops.append(1)
        
        adj_count = 0
        for edge in graph.edges:
            if edge[0] == face_idx or edge[1] == face_idx:
                adj_count += 1
        face_adjs.append(adj_count)
    
    # 转换为numpy数组
    graph_face_feat = np.asarray(graph_face_feat)
    face_types = np.array(face_types)
    face_areas = np.array(face_areas, dtype=np.float32)
    face_loops = np.array(face_loops)
    face_adjs = np.array(face_adjs)

    # 准备边特征列表 - 修正为7通道
    graph_edge_feat = []    # U网格几何特征 [n, 7]
    edge_types = []         # 边类型
    edge_lengths = []       # 边长度
    edge_angles = []        # 边角度
    edge_convs = []         # 边凸度
    
    # 处理每条边
    for edge_idx in graph.edges:
        edge = graph.edges[edge_idx]["edge"]
        
        # 忽略没有曲线的边（如圆锥顶点）
        if not edge.has_curve():
            # 添加默认值
            edge_types.append(0)
            edge_lengths.append(0.0)
            edge_angles.append(0.0)
            edge_convs.append(0)
            continue
        
        # U网格采样 - 6通道特征
        points = ugrid(edge, method="point", num_u=curv_num_u_samples)
        tangents = ugrid(edge, method="tangent", num_u=curv_num_u_samples)
        
        # 修正：将6通道扩展为7通道
        # 拼接成6通道特征 [n, 6] - XYZ(3) + 切线(3)
        edge_feat_6ch = np.concatenate((points, tangents), axis=-1)
        
        # # 添加第7个通道（零填充）
        # zero_channel = np.zeros((edge_feat_6ch.shape[0], 1), dtype=np.float32)
        # edge_feat = np.concatenate((edge_feat_6ch, zero_channel), axis=-1)
        # 添加第7个通道（一填充）- 使用np.full
        one_channel = np.full((edge_feat_6ch.shape[0], 1), 1.5707963705062866, dtype=np.float32)
        edge_feat = np.concatenate((edge_feat_6ch, one_channel), axis=-1)
        
        graph_edge_feat.append(edge_feat)
        
        # 获取边类型
        try:
            curve_type = str(edge.curve_type()).lower()
            edge_type = EDGE_TYPE_MAP.get(curve_type, 0)
        except:
            edge_type = 0
        edge_types.append(edge_type)
        
        # 计算边长度
        try:
            start_point = edge.start_point()
            end_point = edge.end_point()
            length = np.linalg.norm(np.array(end_point) - np.array(start_point))
            edge_lengths.append(length)
        except:
            edge_lengths.append(0.0)
        
        # 计算边角度（简化版）
        try:
            if curv_num_u_samples >= 2:
                start_tangent = tangents[0]
                end_tangent = tangents[-1]
                start_tangent_norm = start_tangent / (np.linalg.norm(start_tangent) + 1e-10)
                end_tangent_norm = end_tangent / (np.linalg.norm(end_tangent) + 1e-10)
                angle = np.arccos(np.clip(np.dot(start_tangent_norm, end_tangent_norm), -1.0, 1.0))
                edge_angles.append(angle)
            else:
                edge_angles.append(0.0)
        except:
            edge_angles.append(0.0)
        
        # 计算边凸度（简化版）
        try:
            edge_convs.append(CONVEXITY_MAP["convex"])
        except:
            edge_convs.append(1)
    
    # 转换为numpy数组
    edge_types = np.array(edge_types, dtype=np.int32)
    edge_lengths = np.array(edge_lengths, dtype=np.float32)
    edge_angles = np.array(edge_angles, dtype=np.float32)
    edge_convs = np.array(edge_convs, dtype=np.int32)
    
    # 处理空的边特征
    if len(graph_edge_feat) == 0:
        graph_edge_feat = np.array([], dtype=np.float32)
    else:
        graph_edge_feat = np.asarray(graph_edge_feat)

    # 创建DGL图
    edges = list(graph.edges)
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    num_nodes = len(graph.nodes)
    dgl_graph = dgl.graph((src, dst), num_nodes=num_nodes)
    
    # 添加节点特征（BrepMFR要求的字段名）
    dgl_graph.ndata["x"] = torch.from_numpy(graph_face_feat).float()  # UV网格特征
    dgl_graph.ndata["z"] = torch.from_numpy(face_types).int()        # 面类型
    dgl_graph.ndata["y"] = torch.from_numpy(face_areas).float()       # 面面积
    dgl_graph.ndata["l"] = torch.from_numpy(face_loops).int()         # 环数量
    dgl_graph.ndata["a"] = torch.from_numpy(face_adjs).int()          # 相邻面数量
    dgl_graph.ndata["f"] = torch.zeros(num_nodes, dtype=torch.int)    # 标签特征
    
    # 添加边特征（BrepMFR要求的字段名）- 修正为7通道
    if len(graph_edge_feat) > 0:
        num_edges = len(graph_edge_feat)
        num_samples = curv_num_u_samples
        
        # 确保格式为 [num_edges, channels=7, num_samples]
        edge_data = torch.zeros((num_edges, num_samples, 7), dtype=torch.float)
        
        for i, feat in enumerate(graph_edge_feat):
            # 确保特征有7个通道
            if feat.shape[-1] < 7:
                # 如果通道数不足，用零填充
                padding = np.zeros((feat.shape[0], 7 - feat.shape[-1]), dtype=np.float32)
                feat = np.concatenate((feat, padding), axis=-1)
            
            # 转置为 [7, num_samples]
            edge_data[i] = torch.from_numpy(feat)
        
        dgl_graph.edata["x"] = edge_data
    
    dgl_graph.edata["t"] = torch.from_numpy(edge_types).int()        # 边类型
    dgl_graph.edata["l"] = torch.from_numpy(edge_lengths).float()     # 边长度
    dgl_graph.edata["a"] = torch.from_numpy(edge_angles).float()      # 边角度
    dgl_graph.edata["c"] = torch.from_numpy(edge_convs).int()        # 边凸度
    
    # 添加图元数据（BrepMFR要求的）
    dgl_graph.gdata = {}
    
    # 1. edges_path - [num_nodes, num_nodes, max_dist]
    max_dist = 16
    edges_path = np.zeros((num_nodes, num_nodes, max_dist), dtype=np.int32)
    
    for i, edge in enumerate(edges):
        u, v = edge
        edges_path[u, v, 0] = i + 1
        edges_path[v, u, 0] = i + 1
    
    for i in range(num_nodes):
        edges_path[i, i, 0] = 0
    
    dgl_graph.gdata["edges_path"] = torch.from_numpy(edges_path).int()
    
    # 2. spatial_pos - 基于面质心的空间位置
    centroids = []
    for face_idx in graph.nodes:
        face = graph.nodes[face_idx]["face"]
        try:
            centroid = face.mid_point()
            centroids.append(centroid)
        except:
            centroids.append([0.0, 0.0, 0.0])
    
    spatial_pos = np.zeros((num_nodes, num_nodes), dtype=np.int32)
    for i in range(num_nodes):
        for j in range(num_nodes):
            distance = np.linalg.norm(np.array(centroids[i]) - np.array(centroids[j]))
            spatial_pos[i, j] = int(distance * 1000)
    
    dgl_graph.gdata["spatial_pos"] = torch.from_numpy(spatial_pos).int()
    
    # 3. d2_distance 和 angle_distance（填充默认值）
    dgl_graph.gdata["d2_distance"] = torch.zeros(num_nodes, num_nodes, 64, dtype=torch.float)
    dgl_graph.gdata["angle_distance"] = torch.zeros(num_nodes, num_nodes, 64, dtype=torch.float)
    
    return dgl_graph

def save_to_binary(graph, filename):
    """
    将图保存为二进制文件，兼容BrepMFR
    
    参数:
        graph: DGL图对象
        filename: 输出文件路径
    """
    metadata = {}
    
    # 从gdata提取元数据
    if hasattr(graph, "gdata"):
        metadata["edges_path"] = graph.gdata.get("edges_path", torch.tensor([]))
        metadata["spatial_pos"] = graph.gdata.get("spatial_pos", torch.tensor([]))
        metadata["d2_distance"] = graph.gdata.get("d2_distance", torch.tensor([]))
        metadata["angle_distance"] = graph.gdata.get("angle_distance", torch.tensor([]))
    else:
        metadata["edges_path"] = torch.tensor([])
        metadata["spatial_pos"] = torch.tensor([])
        metadata["d2_distance"] = torch.tensor([])
        metadata["angle_distance"] = torch.tensor([])
    
    # 使用DGL的保存函数
    dgl.data.utils.save_graphs(filename, [graph], metadata)
    
    # 保持向后兼容性
    graph.data = metadata
    graph.gdata = metadata

def validate_graph(graph):
    """
    验证图是否包含BrepMFR所需的所有字段
    
    参数:
        graph: DGL图对象
    
    返回:
        bool: 验证是否通过
    """
    required_node_fields = ["x", "z", "y", "l", "a", "f"]
    required_edge_fields = ["t", "l", "a", "c", "x"]
    required_graph_fields = ["edges_path", "spatial_pos", "d2_distance", "angle_distance"]
    
    # 验证节点字段
    for field in required_node_fields:
        if field not in graph.ndata:
            print(f"警告: 缺少节点字段: {field}")
            return False
    
    # 验证边字段
    if len(graph.edges()) > 0:
        for field in required_edge_fields:
            if field not in graph.edata:
                print(f"警告: 缺少边字段: {field}")
                return False
    
    # 验证图元数据
    if hasattr(graph, "gdata"):
        for field in required_graph_fields:
            if field not in graph.gdata:
                print(f"警告: 缺少图元数据字段: {field}")
                return False
    
    return True

def convert_stp_to_bin(stp_file_path, output_bin_path=None, curv_u_samples=5, surf_u_samples=5, surf_v_samples=5):
    """
    STEP到BIN文件转换接口
    
    参数:
        stp_file_path: 输入STEP文件路径
        output_bin_path: 输出BIN文件路径（可选）
        curv_u_samples: 曲线采样点数（默认5）
        surf_u_samples: 曲面u方向采样点数（默认5）
        surf_v_samples: 曲面v方向采样点数（默认5）
    
    返回:
        str: 生成的BIN文件路径
    """
    filename = pathlib.Path(stp_file_path).stem
        
    if output_bin_path is None:
        output_bin_path = str(pathlib.Path(stp_file_path).with_suffix('.bin'))

    pathlib.Path(output_bin_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        # 检查文件是否存在，如果存在则打印警告但仍继续处理
        if pathlib.Path(output_bin_path).exists():
            print(f"警告: 文件已存在，将覆盖: {output_bin_path}")
            # 不返回，继续执行以覆盖文件
        
        print(f"转换文件: {stp_file_path}")
        solids = load_step(stp_file_path)
        if not solids or len(solids) == 0:
            print(f"警告: 文件中未找到实体: {stp_file_path}")
            return None
        
        solid = solids[0]
        
        graph = build_graph(
            solid, curv_u_samples, surf_u_samples, surf_v_samples
        )
        
        if not validate_graph(graph):
            print(f"警告: 图验证失败: {stp_file_path}")
        
        save_to_binary(graph, output_bin_path)
        print(f"成功转换: {output_bin_path}")
        return output_bin_path
        
    except Exception as e:
        print(f"转换错误 {stp_file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_test_graph():
    """创建测试图"""
    # 创建简单图结构
    src = [0, 1, 2]
    dst = [1, 2, 0]
    dgl_graph = dgl.graph((src, dst), num_nodes=3)
    
    # 添加节点特征
    dgl_graph.ndata["x"] = torch.zeros(3, 5, 5, 7)  # UV网格特征 [3, 5, 5, 7]
    dgl_graph.ndata["z"] = torch.tensor([0, 1, 2], dtype=torch.int)  # 面类型
    dgl_graph.ndata["y"] = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float)  # 面面积
    dgl_graph.ndata["l"] = torch.tensor([1, 1, 1], dtype=torch.int)  # 环数量
    dgl_graph.ndata["a"] = torch.tensor([2, 2, 2], dtype=torch.int)  # 相邻面数量
    dgl_graph.ndata["f"] = torch.zeros(3, dtype=torch.int)  # 标签特征
    
    # 添加边特征
    dgl_graph.edata["x"] = torch.zeros(3, 5, 7).type(FloatTensor)  # U网格特征 [3, 5, 7] 修正维度以匹配模型期望
    dgl_graph.edata["t"] = torch.tensor([0, 1, 2], dtype=torch.int)  # 边类型
    dgl_graph.edata["l"] = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float)  # 边长度
    dgl_graph.edata["a"] = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float)  # 边角度
    dgl_graph.edata["c"] = torch.tensor([1, 1, 1], dtype=torch.int)  # 边凸度
    
    # 添加图元数据
    dgl_graph.gdata = {}
    
    # edges_path [3, 3, 16]
    max_dist = 16
    num_nodes = 3
    edges_path = torch.zeros((num_nodes, num_nodes, max_dist), dtype=torch.int)
    edges_path[0, 1, 0] = 1
    edges_path[1, 2, 0] = 2
    edges_path[2, 0, 0] = 3
    edges_path[1, 0, 0] = 1
    edges_path[2, 1, 0] = 2
    edges_path[0, 2, 0] = 3
    for i in range(num_nodes):
        edges_path[i, i, 0] = 0
    dgl_graph.gdata["edges_path"] = edges_path
    
    # spatial_pos [3, 3]
    dgl_graph.gdata["spatial_pos"] = torch.zeros(3, 3, dtype=torch.int)
    
    # d2_distance 和 angle_distance [3, 3, 64]
    dgl_graph.gdata["d2_distance"] = torch.zeros(3, 3, 64, dtype=torch.float)
    dgl_graph.gdata["angle_distance"] = torch.zeros(3, 3, 64, dtype=torch.float)
    
    return dgl_graph

def test_save_load():
    """测试保存和加载功能"""
    # 创建测试目录
    test_dir = os.path.join(os.getcwd(), "test_output")
    os.makedirs(test_dir, exist_ok=True)
    
    # 创建测试文件路径
    test_file = os.path.join(test_dir, "test_graph_001.bin")
    
    # 创建测试图
    graph = create_test_graph()
    
    # 保存到二进制
    save_to_binary(graph, test_file)
    print(f"测试图已保存到: {test_file}")
    
    # 测试加载图
    try:
        graphs, metadata = dgl.data.utils.load_graphs(test_file)
        loaded_graph = graphs[0]
        
        # 检查节点字段
        required_node_fields = ["x", "z", "y", "l", "a", "f"]
        for field in required_node_fields:
            if field in loaded_graph.ndata:
                print(f"✓ 节点字段 '{field}' 存在: 形状={loaded_graph.ndata[field].shape}")
            else:
                print(f"✗ 节点字段 '{field}' 缺失!")
        
        # 检查边字段
        required_edge_fields = ["t", "l", "a", "c", "x"]
        for field in required_edge_fields:
            if field in loaded_graph.edata:
                print(f"✓ 边字段 '{field}' 存在: 形状={loaded_graph.edata[field].shape}")
            else:
                print(f"✗ 边字段 '{field}' 缺失!")
        
        # 检查图元数据
        required_graph_fields = ["edges_path", "spatial_pos", "d2_distance", "angle_distance"]
        for field in required_graph_fields:
            if field in metadata:
                print(f"✓ 图元数据字段 '{field}' 存在: 形状={metadata[field].shape}")
            else:
                print(f"✗ 图元数据字段 '{field}' 缺失!")
        
        print(f"测试完成成功。图文件位于: {test_file}")
        return test_file
    
    except Exception as e:
        print(f"测试期间出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def process_one_file_enhanced(arguments):
    """
    增强版单文件处理函数，改进文件名处理和错误恢复
    """
    fn, args = arguments
    fn_stem = fn.stem
    
    print(f"处理文件: {fn}")
    output_path = pathlib.Path(args.output)
    
    # 确保输出目录存在
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 更精确的文件名处理逻辑
    if fn_stem.isdigit():  # 如果文件名完全由数字组成
        output_file = str(output_path.joinpath(f"{fn_stem}.bin"))
    else:
        # 查找文件名中的数字序列
        import re
        numbers = re.findall(r'\d+', fn_stem)
        if numbers:
            # 使用最后一个数字序列作为标识
            output_file = str(output_path.joinpath(f"{fn_stem}.bin"))
        else:
            # 完全没有数字，添加序列号
            output_file = str(output_path.joinpath(f"{fn_stem}_001.bin"))
    
    try:
        # 检查文件是否已存在（可选：添加覆盖选项）
        if pathlib.Path(output_file).exists() and not getattr(args, 'overwrite', False):
            print(f"文件已存在，跳过: {output_file}")
            return output_file
        
        # 执行转换
        result = convert_stp_to_bin(
            str(fn), 
            output_file, 
            args.curv_u_samples, 
            args.surf_u_samples, 
            args.surf_v_samples
        )
        
        return result
        
    except Exception as e:
        print(f"处理文件 {fn} 时出错: {str(e)}")
        # 返回None表示失败，但继续处理其他文件
        return None

def initializer():
    """初始化函数，忽略子进程中的中断信号"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def process_enhanced(args):
    """
    增强版处理函数，支持更多文件类型和更好错误处理
    """
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    
    # 匹配多种STEP文件扩展名
    step_patterns = ["*.stp", "*.step", "*.STP", "*.STEP"]
    step_files = []
    for pattern in step_patterns:
        step_files.extend(input_path.glob(pattern))
        
    # print(step_files)
    
    # 去重（某些模式可能匹配相同文件）
    step_files = list(set(step_files))
    
    if not step_files:
        print(f"在目录中未找到STEP文件: {input_path}")
        return
    
    print(f"找到 {len(step_files)} 个STEP文件")
    
    pool = Pool(processes=args.num_processes, initializer=initializer)
    
    successful = 0
    failed = 0
    
    try:
        # 使用tqdm显示进度
        with tqdm(total=len(step_files), desc="处理进度") as pbar:
            for result in pool.imap(process_one_file_enhanced, zip(step_files, repeat(args))):
                pbar.update(1)
                if result is not None:
                    successful += 1
                else:
                    failed += 1
                pbar.set_postfix({'成功': successful, '失败': failed})
    
    except KeyboardInterrupt:
        print("\n用户中断处理...")
        pool.terminate()
        pool.join()
        return
    
    finally:
        pool.close()
        pool.join()
    
    print(f"处理完成: {successful} 个成功, {failed} 个失败")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="将实体模型转换为带UV网格特征的面向接图"
    )
    parser.add_argument("input", type=str, nargs='?', help="STEP文件输入目录")
    parser.add_argument("output", type=str, nargs='?', help="DGL图BIN文件输出目录")
    parser.add_argument("--input_file", type=str, help="单个STEP文件转换")
    parser.add_argument("--output_file", type=str, help="单个文件输出路径")
    parser.add_argument("--curv_u_samples", type=int, default=5, help="曲线采样点数")
    parser.add_argument("--surf_u_samples", type=int, default=5, help="曲面u方向采样点数")
    parser.add_argument("--surf_v_samples", type=int, default=5, help="曲面v方向采样点数")
    parser.add_argument("--num_processes", type=int, default=8, help="进程数")
    parser.add_argument("--test", action="store_true", help="生成测试图")
    
    args = parser.parse_args()
    
    if args.test:
        test_save_load()
    elif args.input_file:
        output_file = args.output_file
        if output_file is None:
            fn_stem = pathlib.Path(args.input_file).stem
            if not any(char.isdigit() for char in fn_stem):
                output_file = str(pathlib.Path(args.input_file).with_name(f"{fn_stem}_001.bin"))
            else:
                output_file = str(pathlib.Path(args.input_file).with_suffix(".bin"))
        
        convert_stp_to_bin(
            args.input_file,
            output_file,
            args.curv_u_samples,
            args.surf_u_samples,
            args.surf_v_samples
        )
    elif args.input and args.output:
        process_enhanced(args)
    else:
        print("错误: 请提供 --test、--input_file 或输入输出目录")
        parser.print_help()

if __name__ == "__main__":
    main()
    
    
# python stp_to_bin.py --input_file /workspace/BrepMFR/input/24-0000000-00983.stp --output_file /workspace/BrepMFR/output/24-0000000-00983.bin

# python stp_to_bin.py /workspace/BrepMFR/mfr_data/step /workspace/BrepMFR/mfr_data/bin --num_processes 4