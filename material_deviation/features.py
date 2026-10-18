import sys, cv2, time, threading, queue, os, json
import numpy as np
from .comparison import encode_reference_visual, decode_reference_visual

# ===================== SuperPoint特征提取器 =====================
class SuperPointFeatureExtractor:
    def __init__(self, weights_path=None, cuda=False):

        try:
            from .superpoint_model import SuperPointFrontend
            SP_AVAILABLE = True
            print("SuperPoint模块导入成功")
        except ImportError as e:
            print(f"无法导入SuperPoint模块: {e}")
            print("将使用ORB作为替代")
            SP_AVAILABLE = False

        """初始化SuperPoint特征提取器"""
        self.weights_path = weights_path
        self.cuda = cuda

        print(f"初始化特征提取器 - SP_AVAILABLE: {SP_AVAILABLE}, weights_path: {weights_path}")

        try:
            if SP_AVAILABLE and weights_path and os.path.exists(weights_path):
                # 使用真正的SuperPointFrontend
                self.superpoint = SuperPointFrontend(
                    weights_path=weights_path,
                    nms_dist=4,
                    conf_thresh=0.03,
                    nn_thresh=0.8,
                    cuda=cuda
                )
                self.available = True
                self.use_orb = False
                # SuperPoint使用L2距离匹配器
                self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                print(f"[OK] SuperPoint特征提取器初始化成功 (使用模型: {os.path.basename(weights_path)})")
            else:
                # 如果SuperPoint不可用或没有权重文件，使用ORB作为替代
                reason = ""
                if not SP_AVAILABLE:
                    reason = "SuperPoint模块不可用"
                elif not weights_path:
                    reason = "未提供权重文件路径"
                elif not os.path.exists(weights_path):
                    reason = f"权重文件不存在: {weights_path}"

                print(f"[WARN] {reason}，使用ORB作为替代")
                self.orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)
                # ORB使用汉明距离匹配器
                self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                self.available = True
                self.use_orb = True
                print("[OK] ORB特征提取器初始化成功（作为SuperPoint替代）")

        except Exception as e:
            print(f"[ERROR] 特征提取器初始化失败: {e}")
            import traceback
            traceback.print_exc()
            # 使用ORB作为后备
            self.orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)
            self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            self.available = True
            self.use_orb = True
            print("[OK] 使用ORB作为后备特征提取器")

    def extract_superpoint_features(self, gray_img, offset_x=0, offset_y=0):
        """使用SuperPoint提取特征点"""
        # 确保图像是float32类型且值在[0,1]范围内
        if gray_img.dtype != np.float32:
            gray_img = gray_img.astype(np.float32) / 255.0

        print(
            f"SuperPoint提取: 图像形状={gray_img.shape}, 类型={gray_img.dtype}, 范围=[{gray_img.min():.3f}, {gray_img.max():.3f}]")

        # 运行SuperPoint
        pts, desc, heatmap = self.superpoint.run(gray_img)

        print(f"SuperPoint返回: {pts.shape[1] if pts.shape[1] > 0 else 0} 个点")

        if pts.shape[1] == 0:
            return []

        # 转换为特征点字典列表
        features = []
        for i in range(pts.shape[1]):
            x, y, score = pts[0, i], pts[1, i], pts[2, i]

            # 获取描述子 - SuperPoint描述子是float32类型
            if desc is not None:
                # 确保描述子是float32类型
                descriptor = desc[:, i].astype(np.float32).tolist()
                if i == 0:  # 打印第一个描述子的信息
                    print(f"第一个描述子: 类型={type(descriptor[0]) if descriptor else 'None'}, 长度={len(descriptor)}")
            else:
                descriptor = []

            feature = {
                'x': float(x + offset_x),
                'y': float(y + offset_y),
                'score': float(score),
                'response': float(score),
                'size': 10.0,
                'angle': 0.0,
                'octave': 0,
                'class_id': -1,
                'descriptor': descriptor
            }
            features.append(feature)

        return features

    def extract_orb_features(self, gray_img, offset_x=0, offset_y=0):
        """使用ORB提取特征点（备用）"""
        # 图像预处理
        processed_gray = self.preprocess_image(gray_img)

        # 提取ORB特征点
        keypoints, descriptors = self.orb.detectAndCompute(processed_gray, None)

        if keypoints is None:
            return []

        # 转换特征点格式
        features = []
        if descriptors is not None:
            for kp, desc in zip(keypoints, descriptors):
                feature = {
                    'x': float(kp.pt[0] + offset_x),
                    'y': float(kp.pt[1] + offset_y),
                    'size': float(kp.size),
                    'angle': float(kp.angle),
                    'response': float(kp.response),
                    'octave': int(kp.octave),
                    'class_id': int(kp.class_id),
                    'descriptor': desc.tolist()
                }
                features.append(feature)
        else:
            for kp in keypoints:
                feature = {
                    'x': float(kp.pt[0] + offset_x),
                    'y': float(kp.pt[1] + offset_y),
                    'size': float(kp.size),
                    'angle': float(kp.angle),
                    'response': float(kp.response),
                    'octave': int(kp.octave),
                    'class_id': int(kp.class_id),
                    'descriptor': []
                }
                features.append(feature)

        return features

    def extract_features(self, image, roi_rect=None):
        """提取特征点"""
        if not self.available:
            return None

        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # 如果有ROI，裁剪图像
            if roi_rect:
                x1, y1, x2, y2 = roi_rect
                # 确保ROI在图像范围内
                h, w = gray.shape[:2]
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                # x2/y2 are exclusive slice bounds; the full-frame ROI ends at w/h.
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))

                # 确保ROI有效
                if x2 > x1 and y2 > y1:
                    roi_gray = gray[y1:y2, x1:x2]
                    offset_x, offset_y = x1, y1
                else:
                    roi_gray = gray
                    offset_x, offset_y = 0, 0
                    print(f"无效的ROI，使用全图")
            else:
                roi_gray = gray
                offset_x, offset_y = 0, 0

            # 根据使用的特征提取器类型提取特征点
            if hasattr(self, 'use_orb') and self.use_orb:
                features = self.extract_orb_features(roi_gray, offset_x, offset_y)
                print(f"ORB提取了 {len(features)} 个特征点（作为SuperPoint替代）")
            else:
                features = self.extract_superpoint_features(roi_gray, offset_x, offset_y)
                print(f"SuperPoint提取了 {len(features)} 个特征点")

            return features

        except Exception as e:
            print(f"特征提取失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def preprocess_image(self, gray):
        """图像预处理：增强对比度、去噪"""
        # CLAHE对比度限制直方图均衡
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 高斯模糊去噪
        blur = cv2.GaussianBlur(enhanced, (3, 3), 0)

        # 锐化增强
        sharpened = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)

        return sharpened

    def save_template(self, features, filepath, reference_frame=None, reference_mask=None):
        """保存特征点模板到文件"""
        if not features or len(features) == 0:
            return False

        try:
            use_orb = bool(getattr(self, "use_orb", False))
            feature_type = "ORB(SuperPoint替代)" if use_orb else "SuperPoint"
            # 准备保存的数据
            template_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "feature_type": feature_type,
                "use_orb": use_orb,
                "num_features": len(features),
                "features": features
            }
            if reference_frame is not None and reference_mask is not None:
                template_data["reference_visual"] = encode_reference_visual(
                    reference_frame, reference_mask
                )

            # 保存到JSON文件
            temp_filepath = filepath + ".tmp"
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, indent=2, ensure_ascii=False)

            # 替换原文件
            if os.path.exists(temp_filepath):
                os.replace(temp_filepath, filepath)
                print(f"模板已保存到: {filepath}")
                return True
            else:
                print("保存失败: 临时文件未创建")
                return False

        except Exception as e:
            print(f"保存模板失败: {e}")
            return False

    def load_template(self, filepath):
        """从文件加载特征点模板"""
        if not os.path.exists(filepath):
            print(f"模板文件不存在: {filepath}")
            return None

        try:
            print(f"开始加载模板: {filepath}")

            with open(filepath, 'r', encoding='utf-8') as f:
                template_data = json.load(f)

            print(f"JSON解析完成")

            # 验证模板数据
            if "features" not in template_data:
                print("模板文件格式错误: 缺少features字段")
                return None

            features = template_data["features"]
            feature_type = template_data.get("feature_type", "SuperPoint")
            num_features = template_data.get("num_features", 0)
            use_orb = template_data.get("use_orb", str(feature_type).lower().startswith("orb"))

            print(f"模板信息: {num_features}个{feature_type}特征点, use_orb={use_orb}")

            # 检查第一个特征点的描述子
            if features and 'descriptor' in features[0] and features[0]['descriptor']:
                desc_list = features[0]['descriptor']
                print(f"第一个描述子: Python类型={type(desc_list[0]) if desc_list else 'None'}, 长度={len(desc_list)}")

            # 将特征点转换为cv2.KeyPoint对象和描述子
            template_keypoints = []
            template_descriptors = []

            for i, feature in enumerate(features):
                # 创建KeyPoint对象
                kp = cv2.KeyPoint(
                    x=feature['x'],
                    y=feature['y'],
                    size=feature.get('size', 10),
                    angle=feature.get('angle', 0),
                    response=feature.get('response', 1.0),
                    octave=feature.get('octave', 0),
                    class_id=feature.get('class_id', -1)
                )
                template_keypoints.append(kp)

                # 获取描述子，根据保存的类型选择正确的数据类型
                if 'descriptor' in feature and feature['descriptor']:
                    # 根据保存的特征类型选择数据类型
                    if use_orb:
                        descriptor = np.array(feature['descriptor'], dtype=np.uint8)
                    else:
                        descriptor = np.array(feature['descriptor'], dtype=np.float32)
                    template_descriptors.append(descriptor)

                    # 打印第一个描述子的信息
                    if i == 0:
                        print(f"第一个描述子numpy数组: 类型={descriptor.dtype}, 形状={descriptor.shape}")

            # 将描述子列表转换为numpy数组
            if template_descriptors:
                template_descriptors = np.vstack(template_descriptors)
                print(f"描述子数组: 类型={template_descriptors.dtype}, 形状={template_descriptors.shape}")
            else:
                template_descriptors = np.array([])

            print(f"模板加载完成: {len(template_keypoints)} 个关键点")

            return {
                'keypoints': template_keypoints,
                'descriptors': template_descriptors,
                'reference_visual': decode_reference_visual(template_data.get('reference_visual')),
                'metadata': {
                    'feature_type': feature_type,
                    'num_features': num_features,
                    'filepath': filepath,
                    'use_orb': use_orb
                }
            }

        except Exception as e:
            print(f"加载模板失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def match_features(self, scene_image, template_data, roi_rect=None):
        """
        在场景图像中匹配模板特征
        """
        if not self.available or template_data is None:
            return None

        try:
            print(f"\n{'=' * 60}")
            print("开始特征匹配")
            print(f"{'=' * 60}")

            # 获取模板信息
            template_metadata = template_data.get('metadata', {})
            template_type = template_metadata.get('feature_type', 'Unknown')
            template_use_orb = template_metadata.get('use_orb', False)
            template_descriptors = template_data['descriptors']

            print(f"模板: {template_metadata.get('num_features', 'Unknown')}个{template_type}特征点")
            print(f"模板描述子: {template_descriptors.shape} ({template_descriptors.dtype})")
            print(f"当前特征提取器使用ORB: {self.use_orb}")

            # 提取场景图像特征
            scene_features = self.extract_features(scene_image, roi_rect)
            if not scene_features:
                print("场景特征提取失败")
                return None

            # 将场景特征转换为KeyPoint和描述子
            scene_keypoints = []
            scene_descriptors = []

            for feature in scene_features:
                kp = cv2.KeyPoint(
                    x=feature['x'],
                    y=feature['y'],
                    size=feature.get('size', 10),
                    angle=feature.get('angle', 0),
                    response=feature.get('response', 1.0),
                    octave=feature.get('octave', 0),
                    class_id=feature.get('class_id', -1)
                )
                scene_keypoints.append(kp)

                if 'descriptor' in feature and feature['descriptor']:
                    # 场景描述子根据当前使用的特征提取器类型确定
                    if self.use_orb:
                        descriptor = np.array(feature['descriptor'], dtype=np.uint8)
                    else:
                        descriptor = np.array(feature['descriptor'], dtype=np.float32)
                    scene_descriptors.append(descriptor)

            if not scene_descriptors:
                print("场景没有描述子")
                return None

            scene_descriptors = np.vstack(scene_descriptors)
            print(f"场景描述子: {scene_descriptors.shape} ({scene_descriptors.dtype})")

            # 检查描述子维度
            if template_descriptors.shape[1] != scene_descriptors.shape[1]:
                print(
                    f"错误: 描述子维度不匹配! 模板={template_descriptors.shape[1]}, 场景={scene_descriptors.shape[1]}")
                return None

            # 检查数据类型是否匹配
            if template_descriptors.dtype != scene_descriptors.dtype:
                print(f"警告: 数据类型不匹配! 模板={template_descriptors.dtype}, 场景={scene_descriptors.dtype}")

                # 根据模板类型决定如何处理
                if template_use_orb:
                    print("模板是ORB特征，转换场景描述子到uint8")
                    if scene_descriptors.dtype == np.float32:
                        # SuperPoint -> ORB
                        scene_descriptors = (scene_descriptors * 255).clip(0, 255).astype(np.uint8)
                else:
                    print("模板是SuperPoint特征，转换场景描述子到float32")
                    if scene_descriptors.dtype == np.uint8:
                        # ORB -> SuperPoint
                        scene_descriptors = scene_descriptors.astype(np.float32) / 255.0

            print(f"转换后场景描述子: {scene_descriptors.shape} ({scene_descriptors.dtype})")

            # 选择匹配器
            if template_use_orb:
                print("使用ORB匹配器 (NORM_HAMMING)")
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                ratio_threshold = 0.7
            else:
                print("使用SuperPoint匹配器 (NORM_L2)")
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                # 0.90 retained more geometrically consistent pairs than 0.75
                # on the supplied scene; RANSAC below rejects outliers.
                ratio_threshold = 0.90

            print(f"开始KNN匹配 (k=2)...")

            # 使用KNN匹配
            matches = matcher.knnMatch(template_descriptors, scene_descriptors, k=2)

            # 应用Lowe's比率测试
            good_matches = []
            for i, match_pair in enumerate(matches):
                if len(match_pair) < 2:
                    continue
                m, n = match_pair
                if n.distance <= 0:
                    continue
                ratio = m.distance / n.distance

                if ratio < ratio_threshold:
                    good_matches.append(m)
                    if len(good_matches) <= 5:  # 打印前5个匹配的信息
                        print(f"  匹配{i}: 距离={m.distance:.3f}/{n.distance:.3f}, 比率={ratio:.3f} [OK]")

            print(f"匹配结果: 原始匹配 {len(matches)} 个, 筛选后 {len(good_matches)} 个")

            if len(good_matches) < 4:
                print(f"匹配点数量不足: {len(good_matches)} < 4")
                # 打印一些调试信息
                if len(matches) > 0:
                    print("前几个匹配的比率:")
                    for i, match_pair in enumerate(matches[:5]):
                        if len(match_pair) >= 2:
                            m, n = match_pair
                            ratio = m.distance / n.distance
                            print(f"  匹配{i}: 距离={m.distance:.3f}/{n.distance:.3f}, 比率={ratio:.3f}")
                return None

            # 获取匹配点对
            template_pts = np.float32([template_data['keypoints'][m.queryIdx].pt for m in good_matches])
            scene_pts = np.float32([scene_keypoints[m.trainIdx].pt for m in good_matches])

            print(f"使用 {len(good_matches)} 个匹配点计算单应性矩阵...")

            # 使用RANSAC计算单应性矩阵
            H, mask = cv2.findHomography(template_pts, scene_pts, cv2.RANSAC, 5.0)

            if H is None:
                print("计算单应性矩阵失败")
                return None

            # 计算匹配的内点
            inlier_indices = np.flatnonzero(mask.ravel())
            inlier_matches = [good_matches[i] for i in inlier_indices]

            print(f"内点数量: {len(inlier_matches)}")

            # Four correspondences can produce a homography by chance. Require
            # enough spatially consistent points before reporting a measurement.
            if len(inlier_matches) < 6 or len(inlier_matches) / len(good_matches) < 0.5:
                print("几何一致的匹配点不足")
                return None

            # Estimate in-plane rotation from a similarity transform. Taking the
            # angle of H's upper-left entry varies with the image origin under
            # perspective and gave unstable angles for the same pair of photos.
            affine, affine_mask = cv2.estimateAffinePartial2D(
                template_pts[inlier_indices], scene_pts[inlier_indices],
                method=cv2.RANSAC, ransacReprojThreshold=5.0,
            )
            if affine is None or affine_mask is None or int(affine_mask.sum()) < 6:
                print("无法得到可靠的平面旋转")
                return None
            rotation = float(np.degrees(np.arctan2(affine[1, 0], affine[0, 0])))
            template_center = np.mean(
                [kp.pt for kp in template_data['keypoints']], axis=0
            )
            matched_center = affine @ np.array([*template_center, 1.0])
            offset = tuple(float(v) for v in matched_center - template_center)

            print(f"匹配成功!")
            print(f"旋转角度: {rotation:.2f}°, 偏移: ({offset[0]:.1f}, {offset[1]:.1f})")

            return {
                'homography': H,
                'matches': inlier_matches,
                'template_keypoints': template_data['keypoints'],
                'scene_keypoints': scene_keypoints,
                'rotation': rotation,
                'offset': offset,
                'num_matches': len(inlier_matches),
                'success': True
            }

        except Exception as e:
            print(f"特征匹配失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _prepare_features_for_json(self, features):
        """准备特征点数据以便保存为JSON"""
        prepared_features = []
        for feature in features:
            prepared_feature = feature.copy()

            # 处理描述子
            if 'descriptor' in prepared_feature and prepared_feature['descriptor']:
                # 将numpy数组转换为Python列表
                if isinstance(prepared_feature['descriptor'], np.ndarray):
                    prepared_feature['descriptor'] = prepared_feature['descriptor'].tolist()

            prepared_features.append(prepared_feature)

        return prepared_features

    def calculate_transform(self, homography):
        """从单应性矩阵计算旋转角度和偏移"""
        try:
            # 从单应性矩阵分解出旋转矩阵
            # 使用前两列作为旋转矩阵的近似
            R = homography[:2, :2]

            # 计算旋转角度（弧度）
            angle_rad = np.arctan2(R[1, 0], R[0, 0])
            # 转换为角度
            angle_deg = np.degrees(angle_rad)

            # 获取平移（偏移）
            tx = homography[0, 2]
            ty = homography[1, 2]

            return angle_deg, (tx, ty)
        except Exception as e:
            print(f"计算变换失败: {e}")
            return 0.0, (0.0, 0.0)


    def calculate_transform_from_matches(self, homography, template_keypoints, scene_keypoints, matches):
        """从匹配点对计算旋转角度和偏移"""
        try:
            # 1. 计算旋转角度（从单应性矩阵）
            R = homography[:2, :2]
            angle_rad = np.arctan2(R[1, 0], R[0, 0])
            angle_deg = np.degrees(angle_rad)

            # 2. 计算偏移（使用所有匹配点对的平均偏移）
            total_offset_x = 0.0
            total_offset_y = 0.0
            valid_matches = 0

            for match in matches:
                template_pt = template_keypoints[match.queryIdx].pt
                scene_pt = scene_keypoints[match.trainIdx].pt

                # 计算这个匹配点对的偏移
                offset_x = scene_pt[0] - template_pt[0]
                offset_y = scene_pt[1] - template_pt[1]

                total_offset_x += offset_x
                total_offset_y += offset_y
                valid_matches += 1

            if valid_matches > 0:
                avg_offset_x = total_offset_x / valid_matches
                avg_offset_y = total_offset_y / valid_matches

                print(f"使用 {valid_matches} 个匹配点计算平均偏移")
                return angle_deg, (float(avg_offset_x), float(avg_offset_y))
            else:
                return angle_deg, (0.0, 0.0)

        except Exception as e:
            print(f"从匹配计算变换失败: {e}")
            return 0.0, (0.0, 0.0)

    def merge_features_by_clustering(self, all_features, distance_threshold=5.0):
        """通过聚类合并特征点"""

        # 收集所有特征点和描述子
        all_points = []
        all_descriptors = []
        all_responses = []

        for frame_data in all_features:
            points = frame_data['points']
            descriptors = frame_data['descriptors']

            for i in range(len(points)):
                all_points.append(points[i])
                all_descriptors.append(descriptors[i])
                all_responses.append(1.0)  # 默认响应值

        if len(all_points) == 0:
            return []

        all_points = np.array(all_points)
        all_descriptors = np.array(all_descriptors)

        print(f"开始聚类合并: {len(all_points)} 个原始特征点")

        # 使用DBSCAN进行聚类（基于位置）
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler

        # 标准化位置数据
        scaler = StandardScaler()
        scaled_points = scaler.fit_transform(all_points)

        # 应用DBSCAN聚类
        # eps: 距离阈值，min_samples: 最小样本数
        dbscan = DBSCAN(eps=0.3, min_samples=2, metric='euclidean')
        labels = dbscan.fit_predict(scaled_points)

        # 统计聚类结果
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # 去除噪声点
        n_noise = list(labels).count(-1)

        print(f"聚类结果: {n_clusters} 个聚类, {n_noise} 个噪声点")

        # 合并每个聚类内的特征点
        merged_features = []

        for label in unique_labels:
            if label == -1:  # 跳过噪声点
                continue

            # 获取当前聚类中的所有点索引
            cluster_indices = np.where(labels == label)[0]

            if len(cluster_indices) < 2:  # 至少需要2个点才合并
                continue

            # 计算聚类中心（位置平均）
            cluster_points = all_points[cluster_indices]
            center_position = cluster_points.mean(axis=0)

            # 计算平均描述子
            cluster_descriptors = all_descriptors[cluster_indices]
            avg_descriptor = cluster_descriptors.mean(axis=0)

            # L2归一化描述子
            norm = np.linalg.norm(avg_descriptor)
            if norm > 0:
                avg_descriptor = avg_descriptor / norm

            # 计算平均响应值
            cluster_responses = [all_responses[i] for i in cluster_indices]
            avg_response = np.mean(cluster_responses)

            # 添加到合并结果
            merged_features.append({
                'position': center_position,
                'descriptor': avg_descriptor,
                'response': avg_response,
                'cluster_size': len(cluster_indices)
            })

        # 按聚类大小排序（聚类越大的特征点越稳定）
        merged_features.sort(key=lambda x: x['cluster_size'], reverse=True)

        # 限制特征点数量
        max_features = 500
        if len(merged_features) > max_features:
            merged_features = merged_features[:max_features]

        print(f"合并后得到 {len(merged_features)} 个稳定的特征点")
        return merged_features
