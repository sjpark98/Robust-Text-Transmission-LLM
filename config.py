import argparse

def def_param():
    param = dict()

    # API key
    param['api_key'] = "YOUR_OPENAI_API_KEY_HERE"

    ## Test params
    param['num_test'] = 10

    ## Model params
    param['p'] = 0.2
    param['log2M'] = 8
    param['M'] = 2 ** param['log2M'] # The number of maps generated
    param['map_length'] = 20
    param['alpha'] = 100
    param['beta'] = 10
    param['gamma'] = 5
    param['delta'] = 1

    ##  Communication params
    param['bits_per_symbol'] = 2 # QPSK, n-QAM
    param['SNR'] = 10
    param['code_rate'] = 1/2
    param['block_length'] = int((param['log2M'] + (1 - param['p']) * param['map_length'] * 7)/param['code_rate'])

    return param


def get_path(param):
    path = dict()
    data_path = './dataset/'

    path['data_path'] = data_path

    return path


class Config:
    def __init__(self):
        self.param = def_param()

        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--param', type=list, default=self.param)
        self.parser.add_argument('--SNR', type=float, default=self.param['SNR'])
        self.parser.add_argument('--p', type=float, default=self.param['p'])
        self.parser.add_argument('--log2M', type=int, default=self.param['log2M'])
        self.parser.add_argument('--map_length', type=int, default=self.param['map_length'])
        self.parser.add_argument('--code_rate', type=float, default=self.param['code_rate'])

        self.opt, _ = self.parser.parse_known_args()
        args = self.parser.parse_args()

        self.param = args.param
        self.param['SNR'] = args.SNR
        self.param['p'] = args.p
        self.param['log2M'] = args.log2M
        self.param['M'] = 2 ** self.param['log2M']
        self.param['map_length'] = args.map_length
        self.param['code_rate'] = args.code_rate
        self.param['block_length'] = int((self.param['log2M'] + (1 - self.param['p']) * self.param['map_length'] * 6)/self.param['code_rate']) * 5
        self.paths = get_path(param=self.param)

    def print_options(self):
        """Print and save options
                It will print both current options and default values(if different).
                It will save options into a text file / [checkpoints_dir] / opt.txt
                """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(self.opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)


