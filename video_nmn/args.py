import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # Input and Output
    parser.add_argument('--dataset', type=str, default='AGQA', help='one of [AGQA]')
    parser.add_argument('--debug', action='store_true', help='small data for debugging') 
    parser.add_argument('--rgb-path', default=None, type=str, help='Path to video feature files', required=True)
    parser.add_argument('--flow-path', default=None, type=str, help='Path to video feature files')
    parser.add_argument('--str2num-path', type=str, help='AGQA video feature str to num', default='./data/AGQA/video_features/strID2numID.json')
    parser.add_argument('--video-secs-path', type=str, help='Charades video length (in secs)', default='./data/AGQA/video_features/video_secs.json')
    parser.add_argument('--output', default=None, type=str, help='output path of model and params')
    parser.add_argument('--result-filename', default=None, type=str, help='output path evaluation results. None for not saving predictions.')
    parser.add_argument('--num-workers', default=2, type=int, help='')
    parser.add_argument('--vocab-filename', type=str, help='vocab dir', default='./data/AGQA/vocab.json')
    parser.add_argument('--glove-filename', type=str, help='', default='./data/glove.6B.300d.txt')
    parser.add_argument('--train-filename', type=str, help='', default='./data/AGQA/train_balanced.pkl')
    parser.add_argument('--valid-filename', type=str, help='', default='./data/AGQA/valid_balanced.pkl')
    parser.add_argument('--test-filename', type=str, help='', default='./data/AGQA/test_balanced.pkl')
    parser.add_argument('--use-prog-word-embeddings', action='store_true',
                        help='if a word in nmn_program is not found in question, use text_encoder to encode it, instead of using all tokens in question')

    # Model
    parser.add_argument('--model-ckpt', default=None, type=str, help='')
    parser.add_argument('--config-filename', default=None, type=str, help='')
    parser.add_argument('--hidden-size', default=512, type=int, help='')
    parser.add_argument('--video-size', default=2048, type=int, help='')
    parser.add_argument('--text-size', default=300, type=int, help='')
    parser.add_argument('--max-video-length', default=150, type=int, help='')
    parser.add_argument('--dropout', default=0.25, type=float, help='dropout rate')
    parser.add_argument('--init-method', type=str, default='default',
                        choices=['xavier_uniform', 'xavier_normal', 'normal', 'default'])
    parser.add_argument('--layer-norm', type=int, default=1)

    # Training
    parser.add_argument('--num-epochs', default=10, type=int, help='Number of epochs')
    parser.add_argument('--rand-seed', default=1, type=int, help="seed for generating random numbers")
    parser.add_argument('--report-interval', default=1000, type=int, help='report interval to log training results')
    parser.add_argument('--evaluate-interval', default=200000, type=int, help='interval to evaluate and save models')
    parser.add_argument('--gradient-accumulation', default=32, type=int, help='')
    parser.add_argument('--lr', default=2e-4, type=float, help='')
    parser.add_argument('--weight-decay', default=0, type=float, help='L2 regularization')
    parser.add_argument('--scheduler-start-factor', type=float, default=1)
    parser.add_argument('--scheduler-end-factor', type=float, default=0.1)
    parser.add_argument('--scheduler-total-iters', type=float, default=200000)

    # ablation study for generalization
    parser.add_argument('--novel-comp', type=int, default=None, help='novel composition generalization (0 or 1). None for load all data')
    parser.add_argument('--more-steps', type=int, default=None, help='more steps generalization (0 or 1). None for load all data')

    # Train by Modules
    parser.add_argument('--train-sg-filename', type=str, help='scene graph filename, for training seperate modules', default=None)
    parser.add_argument('--valid-sg-filename', type=str, help='scene graph filename, for training seperate modules', default=None)
    parser.add_argument('--test-sg-filename', type=str, help='scene graph filename, for training seperate modules', default=None)
    parser.add_argument('--id2word-filename', type=str, help='ENG.txt filename, for training seperate modules', default=None)
    parser.add_argument('--word2id-filename', type=str, help='IDX.txt filename, for training seperate modules', default=None)
    parser.add_argument('--module-loss-weight', type=float, default=1.0, help='the weight of module loss')
    parser.add_argument('--decoder-loss-weight', type=float, default=1.0, help='the weight of decoder loss')
    parser.add_argument('--train-module-before-iters', type=int, default=1e10, help='use no decoder loss before x iterations')
    parser.add_argument('--train-decoder-after-iters', type=int, default=0, help='use only decoder loss after x iterations')
    parser.add_argument('--modules-no-intermediate-train', type=str, default=['FilterFrame'], nargs='+', help='modules not to train using the intermediate loss')

    # evaluate
    parser.add_argument('--evaluate-func', type=str, default='acc')
    parser.add_argument('--modules-to-check', nargs='+', type=str, help='used in evaluate.py, modules to run for ckeck_module_output')
    parser.add_argument('--module-to-check', type=str, default='Filter', help='used in evaluate.py, modules to run for check_filter_result')
    parser.add_argument('--start-index', type=int, default=0, help='evaluation start index')
    parser.add_argument('--end-index', type=int, default=-1, help='evaluation end index, -1 for the last example')
    parser.add_argument('--filter-answer-vocab-filename', type=str, default='./data/AGQA/filter_answers.json')

    # using with pre-trained models
    parser.add_argument('--lm-model', type=str, default='VideoGPT')
    parser.add_argument('--bert-path', type=str, default=None)
    parser.add_argument('--llm-lora', action='store_true')
    parser.add_argument('--batch-size', default=16, type=int, help='only used for train & evaluate pre-trained models')
    parser.add_argument('--tokenizer-max-length', default=64, type=int, help='max token length of bert tokenizer')
    parser.add_argument('--gpt-video-loss-weight', type=int, default=1, help='video loss weight when traing LLM as the decoder')
    parser.add_argument('--gpt-max-per-filter-module', type=int, default=1, help='how many retrieved filter module answer per module as gpt prompt')
    parser.add_argument('--gpt-max-filter-output-list-length', type=int, default=5, help='how many retrieved filter module answer in all as gpt prompt')
    parser.add_argument('--gpt-filter-result-path', type=str, default='./snap/20221125-210430-agqa1-my_feat-cossim-cont_nolinear/filter_text_results/%s-result.pkl.%d',
                        help='load gold filter output from scene graph')
    parser.add_argument('--gpt-gold-filter-output', type=int, default=0, help='load gold filter output from scene graph')
    parser.add_argument('--gpt-filter-output-by-level', type=int, default=0,
                        help='if not zero, only use results of filter module <= this level. Overwrites --gpt-max-filter-output-list-length')
    parser.add_argument('--gpt-test', type=int, default=0, help='run test instead of train')


    # video feature tests
    parser.add_argument('--feat-dim-reduce', type=str, default='mean', choices=['mean', 'concat', 'mid'])
    parser.add_argument('--shuffle-video', type=int, default=0, help='use random swapped video features')

    args = parser.parse_args()

    if args.modules_no_intermediate_train is None:
        args.modules_no_intermediate_train = []
    if args.modules_to_check is None:
        args.modules_to_check = []

    return args

