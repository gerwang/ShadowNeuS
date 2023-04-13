class ResolutionScheduler:
    def __init__(self, start_res, res_ckpt, res_val):
        self.cur_res = start_res
        self.res_ckpt = res_ckpt
        self.res_val = res_val
        self.cur_res_idx = 0

    def update_res_level(self, iter_step):
        while self.cur_res_idx < len(self.res_ckpt):
            if self.res_ckpt[self.cur_res_idx] < iter_step:
                self.cur_res = self.res_val[self.cur_res_idx]
                self.cur_res_idx += 1
            else:
                break
        return self.cur_res
