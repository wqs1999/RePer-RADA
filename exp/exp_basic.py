from models import RPTF_Llama, RPTF_Gpt2, RPTF_Opt_1b, RPTF


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'RPTF_Llama': RPTF_Llama,
            'RPTF_Gpt2': RPTF_Gpt2,
            'RPTF_Opt_1b': RPTF_Opt_1b,
            'RPTF': RPTF,
        }
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
