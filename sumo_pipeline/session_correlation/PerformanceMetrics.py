from functools import cached_property


class PerformanceMetrics:
    tp: int
    tn: int
    fp: int
    fn: int

    def __init__(self, missed_client_flows_full_pipeline, missed_os_flows_full_pipeline) -> None:
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = missed_client_flows_full_pipeline + missed_os_flows_full_pipeline
        self.__cached_fpr = None
        self.__cached_fnr = None
        self.__cached_precision = None
        self.__cached_recall = None
        self.__cached_f1_score = None

    def __set_fpr(self):
        if self.fp + self.tn == 0:
            self.__cached_fpr = 0
        else:
            self.__cached_fpr = self.fp / (self.fp + self.tn)

    def __set_fnr(self):
        if self.fn + self.tp == 0:
            self.__cached_fnr = 0
        else:
            self.__cached_fnr = self.fn / (self.fn + self.tp)

    def __set_precision(self):
        if self.tp + self.fp == 0:
            self.__cached_precision = 0
        else:
            self.__cached_precision = self.tp / (self.tp + self.fp)
        
    def __set_recall(self):
        if self.tp + self.fn == 0:
            self.__cached_recall = 0
        else:
            self.__cached_recall = self.tp / (self.tp + self.fn)
                
    def __set_f1_score(self):
        if self.__cached_precision + self.__cached_recall == 0:
            self.__cached_f1_score = 0
        else:
            self.__cached_f1_score = (2 * self.__cached_precision * self.__cached_recall) / (self.__cached_precision + self.__cached_recall)

    @cached_property
    def fpr(self) -> float:
        if self.__cached_fpr is None:
            self.calculate_performance_scores()
        return self.__cached_fpr
    
    @cached_property
    def fnr(self) -> float:
        if self.__cached_fnr is None:
            self.calculate_performance_scores()
        return self.__cached_fnr

    @cached_property
    def precision(self) -> float:
        if self.__cached_precision is None:
            self.calculate_performance_scores()
        return self.__cached_precision
    
    @cached_property
    def recall(self) -> float:
        if self.__cached_recall is None:
            self.calculate_performance_scores()
        return self.__cached_recall
    
    @cached_property
    def f1_score(self) -> float:
        if self.__cached_f1_score is None:
            self.calculate_performance_scores()
        return self.__cached_f1_score

    def calculate_performance_scores(self):
        self.__set_precision()
        self.__set_recall()
        self.__set_fpr()
        self.__set_fnr()
        self.__set_f1_score()

    def __repr__(self):
        return f"""tp: {self.tp}; fp: {self.fp}; tn: {self.tn}; fn: {self.fn}; 
                Precision: {self.precision};
                recall: {self.recall};
                fpr: {self.fpr}; fnr: {self.fnr};
                f1-score: {self.f1_score}\n"""