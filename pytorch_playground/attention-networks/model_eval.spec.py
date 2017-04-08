from train import Session
from utils import ledger, Struct

args = Struct(**{'BATCH_SIZE': 10,
                 'BI_DIRECTIONAL': False,
                 'DASH_ID': 'seq-to-seq-experiment',
                 'DEBUG': True,
                 'EVAL_INTERVAL': 10,
                 'TEACHER_FORCING_R': 0.5,
                 'INPUT_LANG': 'eng',
                 'LEARNING_RATE': 0.001,
                 'MAX_DATA_LEN': 50,
                 'MAX_OUTPUT_LEN': 100,
                 'N_EPOCH': 5,
                 'N_LAYERS': 1,
                 'OUTPUT_LANG': 'cmn',
                 'SAVE_INTERVAL': 100})

with Session(args) as sess:
    for i in range(args.N_EPOCH):
        sess.train()
    # sess.load_pretrain('./trained/test.cp')
    sentence = sess.evaluate('This is a job.')
    print(sentence)

# want to figure out why the execution stops at 1 output
