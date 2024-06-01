import os
from collections import defaultdict
from PIL import Image
import torch
from PIL import ImageDraw
import pickledb

from transformers import OwlViTProcessor, OwlViTForObjectDetection


def return_default_dict_set():
    return defaultdict(set)

def return_default_dict_list():
    return defaultdict(list)

class StructuredMemory(object):
    def __init__(self) -> None:
        self.connectivity = defaultdict(return_default_dict_set)
        self.detection_result = defaultdict(return_default_dict_list)
        self.recorded_viewpoint_with_instr_id = defaultdict(set)
        
        self.detector = None
        self.processor = None

        self.db_filename = 'real_time_detection.db'
        self.real_time_detect_db = pickledb.load(self.db_filename, False)
        self.db_entry_num = len(self.real_time_detect_db.getall())
        # assert self.db_entry_num == 0
        print("Real-time Detections Saving to Database: {}".format(self.db_filename))
        print("Already {} entries in Database".format(self.db_entry_num))

        self.hist_inst_id_this_tour = defaultdict(list)

    def clear_memory(self):
        # self.connectivity = defaultdict(lambda: defaultdict(set))
        self.connectivity = defaultdict(return_default_dict_set)
        # self.detection_result = defaultdict(lambda: defaultdict(list))
        self.detection_result = defaultdict(return_default_dict_list)
        self.recorded_viewpoint_with_instr_id = defaultdict(set)
        self.hist_inst_id_this_tour = defaultdict(list) # scan id to list of inst id

    def get_records_num(self):
        record_num = 0
        for k, v in self.recorded_viewpoint_with_instr_id:
            record_num += len(v)
        return record_num

    def get_neighbours_with_depth(self, scan, viewpoint, depth):
        neighbours = set()
        neighbours_depth = dict()

        neighbours_records = set()
        neighbours_records.add(viewpoint)
        
        neighbours.add((viewpoint, -1))
        neighbours_depth[(viewpoint, -1)] = 0

        neighbours_last = set()
        neighbours_cur = set()
        for depth_index in range(1, depth+1):
            if depth_index == 1:
                neighbours_cur = self.connectivity[scan][viewpoint]
                for nb in neighbours_cur:
                    if nb not in neighbours:
                        if nb[0] not in neighbours_records:
                            neighbours.add(nb)
                            neighbours_depth[nb] = depth_index
                            neighbours_records.add(nb[0])
                neighbours_last = neighbours_cur
                neighbours_cur = set()
            else:
                for nb in neighbours_last:
                    nb_viewpoint, nb_pointId = nb
                    next_neighbours = self.connectivity[scan][nb_viewpoint]
                    # change pointId to nb_pointId
                    next_neighbours = [(item[0], nb_pointId) for item in next_neighbours]
                    neighbours_cur.update(next_neighbours)
                for nb in neighbours_cur:
                    if nb not in neighbours:
                        if nb[0] not in neighbours_records:
                            neighbours.add(nb)
                            neighbours_depth[nb] = depth_index
                            neighbours_records.add(nb[0])
                neighbours_last = neighbours_cur
                neighbours_cur = set()
        
        neighbours = list(neighbours)
        neighbours_depth = [neighbours_depth[_] for _ in neighbours]

        return neighbours, neighbours_depth
    
    def filtering_detection_result_vp(self, detect):
        
        highest_score = dict()
        highest_index = dict()
        for index, (label, score, viewindex, box) in enumerate(detect):
            if label not in highest_score or score > highest_score[label]:
                highest_score[label] = score
                highest_index[label] = index
        new_detect = [detect[i] for i in highest_index.values()]
        return new_detect
    
    def filtering_detection_result_vp_keywords(self, detect, keywords):
        
        highest_score = dict()
        highest_index = dict()
        for index, (label, score, viewindex, box) in enumerate(detect):
            keywords_match = False
            for keyword in keywords:
                if label in keyword or keyword in label:
                    keywords_match = True
                    break
            if keywords_match and (label not in highest_score or score > highest_score[label]):
                highest_score[label] = score
                highest_index[label] = index
        new_detect = [detect[i] for i in highest_index.values()]
        return new_detect

    def filtering_detection_result_vp_detection_pick(self, detect):
        highest_score = dict()
        highest_index = dict()
        for index, (label, score, viewindex, box) in enumerate(detect):
            if label not in highest_score or score > highest_score[label]:
                highest_score[label] = score
                highest_index[label] = index

        sorted_key = sorted(highest_score, key=highest_score.get, reverse=True)[:3]
        new_detect = [detect[highest_index[label]] for label in sorted_key]
        # new_detect = [detect[index] for label, index in highest_index.items() if highest_score[label] > 0.12]
        return new_detect

    def filtering_detection_result_neighbours(self, detect):
        return detect

    def get_detection_result(self, scan, viewpoint, depth=0, filtering=True, filtering_keywords=None, detection_pick=False):
        neighbours, neighbours_depth = self.get_neighbours_with_depth(scan, viewpoint, depth)
        # cur_detect = self.filtering_detection_result(self.detection_result[scan][viewpoint])
        cur_detect = []
        for neighbour in neighbours:
            if filtering:
                if filtering_keywords is None:
                    if detection_pick is False:
                        neighbour_detect = self.filtering_detection_result_vp(self.detection_result[scan][neighbour[0]])
                    else:
                        neighbour_detect = self.filtering_detection_result_vp_detection_pick(self.detection_result[scan][neighbour[0]])
                else:
                    neighbour_detect = self.filtering_detection_result_vp_keywords(self.detection_result[scan][neighbour[0]], filtering_keywords)
            else:
                neighbour_detect = self.detection_result[scan][neighbour[0]]
            cur_detect.append(neighbour_detect)
        # cur_detect = self.filtering_detection_result_neighbours(cur_detect)
        assert len(cur_detect) == len(neighbours_depth)
        return list(zip(neighbours, cur_detect, neighbours_depth))
    
    def get_batch_detection_result(self, scans, viewpoints, neightbours_depth, filtering_keywords=None, detection_pick=False):
        cur_detects = list()
        if filtering_keywords is not None:
            for scan, viewpoint, filtering_keyword in zip(scans, viewpoints, filtering_keywords):
                cur_detect = self.get_detection_result(scan, viewpoint, neightbours_depth, filtering_keywords=filtering_keyword, detection_pick=detection_pick)
                cur_detects.append(cur_detect)
        else:
            for scan, viewpoint in zip(scans, viewpoints):
                cur_detect = self.get_detection_result(scan, viewpoint, neightbours_depth, detection_pick=True)
                cur_detects.append(cur_detect)
        return cur_detects

    def update(self, scans, viewpoints, detections, candidates, instr_ids, pointIds, real_time_detect=False, instr_id_to_milestones=None, use_tour_hist_inst=False, pre_detect_db=None):
        # maintain connectivity
        for scan, viewpoint, candidate, pointId in zip(scans, viewpoints, candidates, pointIds):
            self.connectivity[scan][viewpoint].update(list(zip(candidate, pointId)))
        # maintain detection_result
        for scan, viewpoint, detection, instr_id in zip(scans, viewpoints, detections, instr_ids):
            if detection is None:
                if (not real_time_detect) or instr_id in self.recorded_viewpoint_with_instr_id[(scan, viewpoint)]:
                    continue
                else:
                    key = '{}%{}%{}'.format(scan, viewpoint, instr_id)
                    detection = self.real_time_detect_db.get(key)
                    if not detection or detection is None:
                        keywords = instr_id_to_milestones[instr_id]
                        print('Detecting scan {} viewpoint {} for instruction {} with keywords {}'.format(scan, viewpoint, instr_id, keywords))
                        self._build_detector()
                        detection = self.detect_milestone(scan, viewpoint, self.processor, self.detector, keywords, draw_boxes=False)
                        if detection is None:
                            continue
                        self.dump_detection_result(scan, viewpoint, instr_id, detection)
                    else:
                        pass
                    # continue
            assert len(detection) == 36
            if instr_id in self.recorded_viewpoint_with_instr_id[(scan, viewpoint)]:
                continue
            self.recorded_viewpoint_with_instr_id[(scan, viewpoint)].add(instr_id)
            for viewindex in range(len(detection)):
                _, boxes, scores, labels = detection[viewindex]
                for box, score, label in zip(boxes, scores, labels):
                    self.detection_result[scan][viewpoint].append((label, score, viewindex, box))
            
            if use_tour_hist_inst:
                self.hist_inst_id_this_tour[scan].append(instr_id)
                for hist_inst_id in self.hist_inst_id_this_tour[scan]:
                    if hist_inst_id in self.recorded_viewpoint_with_instr_id[(scan, viewpoint)]:
                        continue
                    detection = self.real_time_detect_db.get('{}%{}%{}'.format(scan, viewpoint, hist_inst_id))
                    if detection is False:
                        detection = list()
                        for ix in range(36):
                            key = '{}%{}%{}%{}'.format(scan, viewpoint, hist_inst_id, ix)
                            result = pre_detect_db.get(key)
                            if result is False:
                                detection = False
                                break
                            else:
                                detection.append(result)
                    if detection is not False:
                        for viewindex in range(len(detection)):
                            _, boxes, scores, labels = detection[viewindex]
                            for box, score, label in zip(boxes, scores, labels):
                                self.detection_result[scan][viewpoint].append((label, score, viewindex, box))
                        self.recorded_viewpoint_with_instr_id[(scan, viewpoint)].add(hist_inst_id)
                    

    def dump_detection_result(self, scan, viewpoint, instr_id, detection):
        key = '{}%{}%{}'.format(scan, viewpoint, instr_id)
        self.real_time_detect_db.set(key, detection)
        self.db_entry_num += 1
        if self.db_entry_num % 100 == 0:
            print('Real Time Database with {} entries'.format(self.db_entry_num))
            self.real_time_detect_db.dump()
    
    def detect_milestone(self, scan_id, viewpoint_id, processor, detector, milestones, draw_boxes=False):
        scan_dir = './datasets/Matterport3D/v1/scans/'
        image_list = list()
        for ix in range(36):
            image_filename = '{}_viewpoint_{}_res_960x720.jpg'.format(viewpoint_id, ix)
            image_path = os.path.join(scan_dir, scan_id, 'matterport_skybox_images', image_filename)
            if not os.path.exists(image_path):
                print("Unpresented Viewpoint: {} {}".format(scan_id, viewpoint_id))
                return None
            image = Image.open(image_path)
            image_list.append(image)
        texts = [milestones for i in range(len(image_list))]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        step_size = 2
        all_keywords = list()
        all_boxes = list()
        all_scores = list()
        all_labels = list()
        for start in range(0, 36, step_size):
            with torch.no_grad():
                inputs = processor(text=texts[start:start+step_size], images=image_list[start:start+step_size], return_tensors="pt").to(device)
                outputs = detector(**inputs)
            
            # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
            target_sizes = torch.Tensor([image.size[::-1] for image in image_list[start:start+step_size]]).to(device)
            # Convert outputs (bounding boxes and class logits) to COCO API
            results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
            # i = 0  # Retrieve predictions for the first image for the corresponding text queries
            # text = texts[i]
            boxes = [result['boxes'].cpu().tolist() for result in results]
            scores = [[score.item() for score in result['scores']] for result in results]
            labels = [[texts[0][label] for label in result['labels']] for result in results] # NOTE: here you have to make sure all texts element are the same.
            all_boxes.extend(boxes)
            all_scores.extend(scores)
            all_labels.extend(labels)
            all_keywords.extend([milestones] * len(labels))

            if draw_boxes:
                for i, image in enumerate(image_list[start:start+step_size]):
                    draw = ImageDraw.Draw(image)
                    draw.text((0, 0), f' '.join(texts[0]), fill='white')

                    for j, label in enumerate(labels[i]):
                        box, score = boxes[i][j], scores[i][j]
                        xmin, ymin, xmax, ymax = box
                        score = score
                        draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
                        draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="white")
                    
                    if len(labels[i]) > 0:
                        image_path = '{}_viewpoint_{}_draw'.format(viewpoint_id, start+i)
                        image.save('./anno_images/' + image_path + '_anno.jpg')

        assert len(all_labels) == 36
        assert len(all_boxes) == 36
        assert len(all_scores) == 36
        assert len(all_keywords) == 36

        # keywords, boxes, scores, labels
        detection = list(zip(all_keywords, all_boxes, all_scores, all_labels))
        return detection
        

    def _build_detector(self):
        if self.processor is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
            self.detector = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14").to(device)