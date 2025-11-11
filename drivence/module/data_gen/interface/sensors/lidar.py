from typing import Dict

import numpy as np


class VirtualLidar:
    def simulate(self, params: Dict) -> np.ndarray:
        """
        输入，雷达需要渲染的参数
        输出，渲染的点云
        """
        raise NotImplementedError("VirtualLidar.simulate must be implemented by subclasses")
