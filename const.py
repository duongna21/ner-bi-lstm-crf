import torch

CHARACTER_LIST = '‘ẹjO:òồọðỡụẽÂưiŨdỜ”ẦVMẩCệ“+v!ỹỉPt2/lứỄỐƠúằíỔẢắDqữ?SJỳẳấwƯ*ẫQơ3éeĩTÔuỲbâựỰÐể0ùừWợBô²èá4êỷÍỖRL,ộ9ynf%g~³¼ẤỗHIG6Ệử°Yậổx(Ởra&Ẩẵễóở)Đm>Áềẻ;sàỮõẬăỵủkỦì]_Ạ7–ÕNãịố8hỪ…\'[Eớ.ạũếXĂờUỒÝỨảpÀầặ1đcýÚ’zÊỏẶoKZ5"FA-'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# CHARACTER_LIST = 'ỏEÓ5UÁẠCẶỸỌù3ủỹ“Ỡỵịsộ>iẨaẢƠúýôẤÐàỐBÌẾữá”Ứ–"ỎỜ‘ŨẺÙYừK2Ẽdâ_XÀÕ:íÉ\'ớỢãơỘ0lửÚ1ỤcỳỖỔÂỲAm[JÔậỡăZỞĩêÊNDHFẲỚ]wểẩố8eSpỂệuPự7ẫõỬ69Ẫèẵẹằồ³ề%&ắẮỰẴẳ~ễqÈỀÃẰợ/4W¼,*tỮ?kở!ổ)-nIógảụéðÒỆòầưỪấLQGẬỊũỒỗĂẦặ²vƯVÝ…Ỉy’ọỄẻìạfđẸbMÍờz°ỷ;(+ứ.ỶĐjĨỦohếxỴRỉẽOTr'

TAG_LIST = ['B-MISC', 'I-MISC', 'B-PER', 'O', 'B-LOC', 'I-PER', 'B-ORG', 'I-ORG', 'I-LOC', '<PAD_TAG>']
POS_LIST = ['R', 'Nc', 'Vy', 'M', 'A', 'Z', 'N', 'CH', 'Ny', 'Nu', 'C', 'V', 'L', 'I', 'P', 'Np', 'T', 'FW', 'X', 'E',
            '<PAD_POS>']
CHUNK_LIST = ['B-VP', 'B-AP', 'B-PP', 'I-AP', 'B-NP', 'O', 'I-VP', 'I-NP', '<PAD_CHUNK>']

NUM_TAGS = len(TAG_LIST) - 1
NUM_POS_TAGS = len(POS_LIST) - 1
NUM_CHUNK_TAGS = len(CHUNK_LIST) - 1

TAG2INDEX = {tag: i for i, tag in enumerate(TAG_LIST)}
POS2INDEX = {pos: i for i, pos in enumerate(POS_LIST)}
CHUNK2INDEX = {chunk: i for i, chunk in enumerate(CHUNK_LIST)}

CHARACTER2INDEX = {character: i for i, character in enumerate(CHARACTER_LIST)}
CHARACTER2INDEX['<UNK>'] = len(CHARACTER2INDEX)
CHARACTER2INDEX['<PAD>'] = len(CHARACTER2INDEX)

TAG_PADDING_IDX = TAG2INDEX['<PAD_TAG>']
POS_PADDING_IDX = POS2INDEX['<PAD_POS>']
CHUNK_PADDING_IDX = CHUNK2INDEX['<PAD_CHUNK>']
