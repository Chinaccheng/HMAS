"""必要模型参数与时间参数；默认值为v11的S2基准，实验变体由experiment_config派生。"""
from dataclasses import asdict,dataclass
from math import isfinite


@dataclass(frozen=True)
class Config:
    n_s:int=24
    n_d:int=12
    n_i:int=24
    region_size:float=100.
    communication_radius:float=25.
    sensing_radius:float=30.
    execution_radius:float=20.
    task_count:int=50
    composite_requirement_count:int=2
    priority_weight:float=10.
    max_hops:int=12
    seed:int=20261026
    search_count:int=300
    validation_search_count:int=1000
    attack_time:int=5
    recovery_start:int=14
    horizon:int=120
    recovery_target:float=.9

    @property
    def node_count(self):return self.n_s+self.n_d+self.n_i

    def validate(self):
        for name in ('n_s','n_d','n_i','task_count','max_hops','search_count','validation_search_count'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<=0:raise ValueError(f'{name} 必须为正整数')
        if self.task_count%5:raise ValueError('任务数必须为5的正整数倍')
        if self.composite_requirement_count not in (2,3):raise ValueError('复合任务固定选择2或3种协同关系')
        if self.max_hops<(6 if self.composite_requirement_count==3 else 5):raise ValueError('max_hops小于任务角色约束的最少跳数')
        for name in ('region_size','communication_radius','sensing_radius','execution_radius'):
            if not isfinite(getattr(self,name)) or getattr(self,name)<=0:raise ValueError(f'{name} 必须为有限正数')
        if not isfinite(self.priority_weight) or self.priority_weight<=1:raise ValueError('priority_weight必须满足1<λ<∞')
        if not 0<self.attack_time<=self.recovery_start<self.horizon:raise ValueError('必须满足0<ta≤tr<T')
        if any(isinstance(v,bool) or not isinstance(v,int) for v in (self.attack_time,self.recovery_start,self.horizon)):
            raise ValueError('时间参数必须为整数步')
        if isinstance(self.seed,bool) or not isinstance(self.seed,int) or self.seed<0:raise ValueError('seed必须为非负整数')
        if not 0<self.recovery_target<=1:raise ValueError('恢复目标必须位于(0,1]')
        return self

    def to_dict(self):return asdict(self)


DEFAULT_CONFIG=Config()
