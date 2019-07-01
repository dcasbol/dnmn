QUESTION_FILE = "data/vqa/Questions/OpenEnded_mscoco_%s_questions.json"
SINGLE_PARSE_FILE = "data/vqa/Questions/%s.sp"
MULTI_PARSE_FILE = "data/vqa/Questions/%s.sps2"
ANN_FILE = "data/vqa/Annotations/mscoco_%s_annotations.json"
IMAGE_FILE = "data/vqa/Images/%s/conv/COCO_%s_%012d.jpg.npz"
RAW_IMAGE_FILE = "data/vqa/Images/%s/raw/COCO_%s_%012d.jpg"
NORMALIZERS_FILE = "data/vqa/Images/normalizers.npz"
INTER_HMAP_FILE = "intermediate/hmaps/{set}/{cat}/{set}-{id}-{cat}.npz"

MAX_ANSWERS = 2000
IMG_DEPTH = 512
HIDDEN_UNITS = 1024
MASK_WIDTH = 14