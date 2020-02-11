MULTI_PARSE_FILE = "multiparse/%s.sps2"
QUESTION_FILE    = "data/vqa/Questions/OpenEnded_mscoco_%s_questions.json"
ANN_FILE         = "data/vqa/Annotations/mscoco_%s_annotations.json"
IMAGE_FILE       = "data/vqa/Images/%s/conv/COCO_%s_%012d.jpg.csr"
RAW_IMAGE_FILE   = "data/vqa/Images/%s/raw/COCO_%s_%012d.jpg"
NORMALIZERS_FILE = "data/vqa/Images/normalizers.npz"
CACHE_HMAP_FILE  = "cache/{set}/hmaps/{set}-hmaps-{qid}.npy"
CACHE_ATT_FILE   = "cache/{set}/attended/{set}-attended-{qid}.npy"
CACHE_QENC_FILE  = "cache/{set}/qenc/{set}-qenc-{qid}.npy"

MAX_ANSWERS    = 2000
IMG_DEPTH      = 512
HIDDEN_UNITS   = 1024
MASK_WIDTH     = 14
VAL_BATCH_SIZE = 512
EMBEDDING_SIZE = 1000
HIDDEN_SIZE    = 500
MAX_VARIANCE   = 0.005