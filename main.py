import argparse
import tensorflow as tf
import os
from model import cyclegan
from ops import *
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Ignore warning


# Define the bool parser for argparse
def str2bool(v: str) -> bool:
    return v.lower() in ('true')

# Define the parser for img shape
def str2shape(s: str) -> tuple:
    try:
        # Split the input string by comma
        s = s.strip().split(',')
        # Check that thi input has only two values
        if len(s) != 2:
            raise argparse.ArgumentTypeError('Image shape must be a tuple of integers')
        return tuple(map(int, s))
    except:
        raise argparse.ArgumentTypeError('Image shape must be a tuple of integers')

# Defien the parser for single image
SINGLE_IMG = ['none', 'std', 'refine', 'hist_spec']
def single_img_parser(v):
    if v not in SINGLE_IMG:
        raise argparse.ArgumentTypeError('Not a valid single image type, please choose from None, std, refine, hist_spec')
    return v


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='alderley', help='path of the dataset used for training')
parser.add_argument('--test_dataset_dir', dest='test_dataset_dir', default='alderley', help='path of the dataset used for testing')
parser.add_argument('--epoch', dest='epoch', type=int, default=20, help='# of epoch')
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=10, help='# of epoch to decay lr')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--img_out_shape', dest='img_out_shape', type=str2shape, default=[256, 1024], help='size of the final output image: height, width')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--n_d', dest='n_d', type=int, default=2, help='# of discriminators')
parser.add_argument('--n_scale', dest='n_scale', type=int, default=2, help='# of scales')
parser.add_argument('--gpu', dest='gpu', type=int, default=0, help='# index of gpu device')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=1000,help='save a model every save_freq iterations')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=100,help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=str2bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./check/alderley/sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./check/alderley/testa2b', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight on L1 term in objective')
parser.add_argument('--use_resnet', dest='use_resnet', type=str2bool, default=True,help='generation network using reidule block')
parser.add_argument('--use_lsgan', dest='use_lsgan', type=str2bool, default=True, help='gan loss defined in lsgan')
parser.add_argument('--max_size', dest='max_size', type=int, default=50, help='max size of image pool, 0 means do not use image pool')
parser.add_argument('--single_img', dest='single_img', type=single_img_parser, default='none', help='save only the converted image')
parser.add_argument('--diag', dest='diag', action='store_true', help='if set while applying the histogram specification, pixel that become brighter will be reverted to original value')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = cyclegan(sess, args)
        show_all_variables()
        model.train(args) if args.phase == 'train' \
            else model.test(args)
if __name__ == '__main__':
    tf.app.run()