from .utils import EvaluationUtil
import re
import threading
from concurrent.futures import ThreadPoolExecutor

# association | aggregation | composition | dependence | generalization | implementation
RELATIONSHIP_LOOKUP_TABLE = {
    "association": {
        "association": 1,    "aggregation": 0.85, "composition": 0.85, "dependence": 0.55, "generalization": 0.17, "implementation": 0.17, 
    },
    "aggregation": {
        "association": 0.85, "aggregation": 1,    "composition": 0.9, "dependence": 0.51, "generalization": 0.17, "implementation": 0.17, 
    },
    "composition": {
        "association": 0.85, "aggregation": 0.9, "composition": 1,    "dependence": 0.51, "generalization": 0.17, "implementation": 0.17, 
    },
    "dependence": {
        "association": 0.55, "aggregation": 0.51, "composition": 0.51, "dependence": 1,    "generalization": 0.46, "implementation": 0.46, 
    },
    "generalization": {
        "association": 0.17, "aggregation": 0.17, "composition": 0.17, "dependence": 0.46, "generalization": 1,    "implementation": 0.72, 
    },
    "implementation": {
        "association": 0.17, "aggregation": 0.17, "composition": 0.17, "dependence": 0.46, "generalization": 0.72, "implementation": 1, 
    }
}

class CLUE:
    # model1 is reference model, and model2 is generated model
    def __init__(self, model1, model2, similarity_func):
        # flexible custom for similarity function
        self.calc_similarity = similarity_func
        # model1 is reference mode; model2 is candidate model;
        self.C1 = model1['classes_info']
        self.C2 = model2['classes_info']
        self.A1 = model1['attributes_info']
        self.A2 = model2['attributes_info']
        self.M1 = model1['methods_info']
        self.M2 = model2['methods_info']
        self.R1 = model1['relationships_info']
        self.R2 = model2['relationships_info']
        # set default parameters
        self.we, self.wr = 0.6, 0.4
        self.wn, self.wa, self.wm = 0.5, 0.3, 0.2
        self.wat, self.wan = 0.4, 0.6
        self.wmn, self.wmt, self.wmp = 0.4, 0.3, 0.3
        self.wpt, self.wpn = 0.4, 0.6
        self.wrt, self.wrq, self.wrn = 0.2, 0.2, 0.6
        # init 
        self.sim_ce, self.sim_ca, self.sim_cm, self.sim_cr,self.clue = 0,0,0,0,0
        # self.match_rc, self.match_rca, self.match_rcm, self.match_rcr = [],[],[],[]
        self.match_rc = []
        self.ES_ELEMENTS = None
        # RunCLUE only once
        self.tik = 1
        # 添加缓存字典来存储计算结果
        self._attr_likeness_cache = {}
        self._method_likeness_cache = {}
        
    
    def RunCLUE(self):
        if self.tik > 0:
            # 第一阶段：获取必要的依赖数据
            self.sim_ce, self.ES_ELEMENTS, self.match_rc, Rflag = self.__class_element_likeness()
            # print("clue-element finished!")
            
            # 第二阶段：创建线程执行并行任务
            # 线程1：处理属性和方法相似度
            def attribute_method_task():
                self.sim_ca = self.calc_class_attribute_likeness(Rflag)
                # print("clue-attribute finished!")
                self.sim_cm = self.calc_class_method_likeness(Rflag)
                # print("clue-method finished!")
            
            # 线程2：处理关系相似度
            def relationship_task():
                self.sim_cr = self.__class_relationship_likeness()
                # print("clue-relationship finished!")
            
            # 创建并启动线程
            thread1 = threading.Thread(target=attribute_method_task)
            thread2 = threading.Thread(target=relationship_task)
            
            thread1.start()
            thread2.start()
            
            # 等待所有线程完成
            thread1.join()
            thread2.join()
            
            # 计算最终结果
            self.clue = self.__CLUE()
            self.tik -= 1
        
    def setCLUEParams(
        self, 
        we: float = 0.6, wr: float = 0.4, # previous setting is we 0.5 wr 0.5
        wn: float = 0.5, wa: float = 0.3, wm: float = 0.2,
        wat: float = 0.4, wan: float = 0.6,
        wmn: float = 0.4, wmt: float = 0.3, wmp: float = 0.3,
        wpt: float = 0.4, wpn: float = 0.6,
        wrt: float = 0.2, wrq: float = 0.2, wrn: float = 0.6 # previous setting is wrt 0.4 wrq 0.2 wrn 0.4
    ):
        if abs((we+wr) - 1)>1e-6: # clue
            raise ValueError("we, wr must sum up to 1")
        if abs((wn+wa+wm) - 1)>1e-6:
            raise ValueError("wn, wa, wm must sum up to 1")
        if abs((wat+wan) - 1)>1e-6: # attribute
            raise  ValueError("wat, wan must sum up to 1")
        if abs((wmn+wmt+wmp) -1)>1e-6: # method
            raise ValueError("wmn, wmt, wmp must sum up to 1")
        if abs((wpt+wpn)-1)>1e-6: # parameter
            raise ValueError("wpt, wpn must sum up to 1")
        if abs((wrt+wrq+wrn) -1)>1e-6: # relationship
            raise ValueError("wrt, wrq, wrn must sum up to 1")
        self.we, self.wr = we, wr
        self.wn, self.wa, self.wm = wn, wa, wm
        self.wat, self.wan = wat, wan
        self.wmn, self.wmt, self.wmp = wmn, wmt, wmp
        self.wpt, self.wpn = wpt, wpn
        self.wrt, self.wrq, self.wrn = wrt, wrq, wrn
    def getCLUE(self):
        return self.clue
    def getES(self):
        return self.ES_ELEMENTS
    def getSimCE(self):
        return self.sim_ce
    def getSimCA(self):
        return self.sim_ca
    def getSimCM(self):
        return self.sim_cm
    def getSimCR(self):
        return self.sim_cr
    def __CLUE(self):
        return self.we*self.sim_ce + self.wr*self.sim_cr
    def __class_element_likeness(self):
        n = len(self.C1)
        m = len(self.C2)
        if n==0: return 1.00, [], [], False
        elif m==0: return 0.00, [], [], False
        else: 
            # 首先计算ES_Names（这个部分也可以并行，但可能收益不大）
            ES_Names = self.calc_similarity([[(self.C1[n_],self.C2[m_]) for m_ in range(m)] for n_ in range(n)])
            
            # 创建ES矩阵的空结构
            ES = [[0.0 for _ in range(m)] for _ in range(n)]
            
            # 定义一个计算单个元素的函数
            def calculate_element(n_, m_):
                return (
                    self.wn * ES_Names[n_][m_]  # class name likeness
                    + self.wa * self.__single_class_attribute_likeness(self.C1[n_], self.C2[m_])
                    + self.wm * self.__single_class_method_likeness(self.C1[n_], self.C2[m_])
                )
            
            # 使用线程池并行计算
            with ThreadPoolExecutor() as executor:
                # 提交所有计算任务
                future_to_indices = {
                    executor.submit(calculate_element, n_, m_): (n_, m_) 
                    for n_ in range(n) for m_ in range(m)
                }
                # 收集结果
                for future in future_to_indices:
                    n_, m_ = future_to_indices[future]
                    ES[n_][m_] = future.result()
            
            # print("ES matrix for class element similarity:")
            # self.__printES(ES)
            
            if n==1 and m==1: return ES[0][0], ES, [0], False
            else: Rflag, best_state, best_energy = EvaluationUtil.find_best_match(n, m, ES)
            
            sim_ce = best_energy / n  # 归一化
            return sim_ce, ES, best_state, Rflag
    def calc_class_attribute_likeness(self, Rflag:bool):
        sim_ca = 0
        for i,j in enumerate(self.match_rc):
            if Rflag:
                c1 = self.C1[j]
                c2 = self.C2[i]
            else:
                c1 = self.C1[i]
                c2 = self.C2[j]
            sim_ca += self.__single_class_attribute_likeness(c1,c2)
        sim_ca = sim_ca / len(self.C1)
        return sim_ca
    def calc_class_method_likeness(self, Rflag:bool):
        sim_cm = 0
        for i,j in enumerate(self.match_rc):
            if Rflag:
                c1 = self.C1[j]
                c2 = self.C2[i]
            else:
                c1 = self.C1[i]
                c2 = self.C2[j]
            sim_cm += self.__single_class_method_likeness(c1,c2)
        sim_cm = sim_cm / len(self.C1)
        return sim_cm
    def __single_class_attribute_likeness(self, c1, c2):
        # 创建缓存键
        cache_key = (c1, c2)
        
        # 检查缓存中是否已有结果
        if cache_key in self._attr_likeness_cache:
            return self._attr_likeness_cache[cache_key]
        
        # 原始计算逻辑
        sim_ca = 0
        a1, a2 = self.A1[c1], self.A2[c2]
        n, m = len(a1), len(a2)
        if n==0 : return 1.00
        elif m==0 : return 0.00
        else:
            ES_Types = self.calc_similarity([[(a1[n_]['type'],a2[m_]['type']) for m_ in range(m)] for n_ in range(n)])
            ES_Names = self.calc_similarity([[(a1[n_]['name'],a2[m_]['name']) for m_ in range(m)] for n_ in range(n)])
            ES = [[
                self.wat*ES_Types[n_][m_]
                + self.wan*ES_Names[n_][m_]
                for m_ in range(m)
            ] for n_ in range(n)]
            if n==1 and m==1: 
                result = ES[0][0]
            else: 
                rflag_, best_state, best_energy = EvaluationUtil.find_best_match(n,m,ES)
                result = best_energy / n  # 归一化
        
        # 存储到缓存
        self._attr_likeness_cache[cache_key] = result
        return result
    
    def __single_class_method_likeness(self, c1, c2):
        # 创建缓存键
        cache_key = (c1, c2)
        
        # 检查缓存中是否已有结果
        if cache_key in self._method_likeness_cache:
            return self._method_likeness_cache[cache_key]
        
        # 原始计算逻辑
        sim_cm = 0
        m1, m2 = self.M1[c1], self.M2[c2]
        n, m = len(m1), len(m2)
        if n==0: return 1.00
        elif m==0: return 0.00
        else:
            ES_Names = self.calc_similarity([[(m1[n_]['name'],m2[m_]['name']) for m_ in range(m)] for n_ in range(n)])
            ES_ReturenTypes = self.calc_similarity([[(m1[n_]['returnType'],m2[m_]['returnType']) for m_ in range(m)] for n_ in range(n)])
            ES = [[
                self.wmn*ES_Names[n_][m_] 
                + self.wmt*ES_ReturenTypes[n_][m_]
                + self.wmp*self.__class_method_param_likeness(m1[n_]['parameters'],m2[m_]['parameters'])
                for m_ in range(m)
            ] for n_ in range(n)]
            if n==1 and m==1: 
                result = ES[0][0]
            else: 
                rflag_, best_state, best_energy = EvaluationUtil.find_best_match(n,m,ES)
                result = best_energy / n  # 归一化
        
        # 存储到缓存
        self._method_likeness_cache[cache_key] = result
        return result
    
    # 如果需要在某些操作后清除缓存（如重新初始化时），可以添加此方法
    def clear_cache(self):
        self._attr_likeness_cache.clear()
        self._method_likeness_cache.clear()
    def __class_method_param_likeness(self, param1:list, param2:list):
        # print(param1,param2)
        n = len(param1)
        m = len(param2)
        if n==0 :return 1.00
        elif m==0:return 0.00
        else:
            ES_Types = self.calc_similarity([[(param1[n_]['type'],param2[m_]['type']) for m_ in range(m)] for n_ in range(n)])
            ES_Names = self.calc_similarity([[(param1[n_]['name'],param2[m_]['name']) for m_ in range(m)] for n_ in range(n)])
            ES = [[
                self.wpt*ES_Types[n_][m_]
                +self.wpn*ES_Names[n_][m_]
                for m_ in range(m)] for n_ in range(n)]
            # print("ES matrix for class method parameter similarity:")
            # self.__printES(ES)
            if n==1 and m==1:return ES[0][0]
            else: rflag_, best_state, best_energy = EvaluationUtil.find_best_match(n,m,ES)
        sim_cp = best_energy / n # 归一化
        return sim_cp
        
    def __get_similarity_from_ES_ELEMENTTS(self,ES,name1,name2):
        n = len(self.C1)
        m = len(self.C2)
        if n==0: return 1.00
        elif m==0:return 0.00
        else:
            name1_index,name2_index=0,0
            for index,name in enumerate(self.C1):
                if name == name1:
                    name1_index=index
                    break
            for index_,name_ in enumerate(self.C2):
                if name_ == name2:
                    name2_index=index_
                    break

            return ES[name1_index][name2_index]  
    def __class_relationship_likeness(self):
        R1,R2 = self.R1, self.R2
        n,m = len(R1), len(R2)
        if n==0 : return 1.00
        elif m==0 : return 0.0
        else:
            ES = [[
                self.wrt*RELATIONSHIP_LOOKUP_TABLE[R1[n_]['r_type']][R2[m_]['r_type']]
                +self.wrq*self.__class_relationship_quantity_likeness(R1[n_],R2[m_])
                +self.wrn*(0.5*self.__get_similarity_from_ES_ELEMENTTS(self.ES_ELEMENTS,R1[n_]['from'],R2[m_]['from']) 
                        + 0.5*self.__get_similarity_from_ES_ELEMENTTS(self.ES_ELEMENTS ,R1[n_]['to'],R2[m_]['to']))
                for m_ in range(m)] for n_ in range(n)]
            # print("ES matrix for class relationship similarity:")
            # self.__printES(ES)
            if n==1 and m==1:return ES[0][0]
            else: rflag_, best_state, best_energy = EvaluationUtil.find_best_match(n,m,ES)
        sim_cr = best_energy / n # 归一化
        return sim_cr
    def __class_relationship_quantity_likeness(self, r1:dict, r2:dict):
        if (r1['r_type'] in ["association", "aggregation", "composition"]) and (r2['r_type'] in ["association", "aggregation", "composition"]):
            r1_q_from,r2_q_from = r1['label']['from'],r2['label']['from']
            r1_q_to,r2_q_to = r1['label']['to'],r2['label']['to']
            p = re.compile(r'\*|many|much|multi')
            likeness = 0
            if (r1_q_from == r2_q_from) or (p.search(r1_q_from) and p.search(r2_q_from)):
                likeness += 0.5
            if (r1_q_to == r2_q_to) or (p.search(r1_q_to) and p.search(r2_q_to)):
                likeness += 0.5
            return likeness
        elif (r1['r_type'] in ["dependence", "generalization", "implementation"]) and (r2['r_type'] in ["dependence", "generalization", "implementation"]):
            return 1.0
        else: return 0.0
    def __printES(self, ES:list):
        for i in range(len(ES)):
            print(f"ES[{i}]:", end=" ")
            for j in range(len(ES[i])):
                print(f"{ES[i][j]}", end=" ")
            print()