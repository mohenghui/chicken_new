


class SplitData:
    root_path = "./0表情质量标注"
    annotation_path = './data/annotations'
    img_path = './data/images'


class SaveImage:
    save_path = "./data/oimages"
    save_edit_path="./data/edit"
    save_classes_path="./data/classes"
    save_angle_path="./data/angle"

class GetGT:
    save_smilar="./data/check"
    predict_dir="./data/yolov7pred"
class DrawCricle:
    classes = ["chin", "eyebrow", "circle", "head", "nose"]

# class WriterXml:
#     detect_class=["circle"]
#     xml_annotation_path="./data/annotations_xml"