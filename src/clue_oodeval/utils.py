import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import torch

class EvaluationUtil:

    MODEL_PATH_BASE = "~/.cache/huggingface/hub"
    MODEL = None

    __TOKENIZER = None
    __MODEL = None

    

    @staticmethod
    def pass_at_k(n, c, k):
        """
        :param n: total number of samples
        :param c: number of correct samples
        :param k: k in pass@$k$
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    @staticmethod
    def find_best_match(n:int, m:int, ES:list):
        # 匈牙利算法求最大匹配
        ES_np = np.array(ES)
        # 匈牙利算法默认是最小化cost，这里取负号变成最大化
        cost_matrix = -ES_np
        if n <= m:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            best_state = col_ind.tolist()
            best_energy = ES_np[row_ind, col_ind].sum()
            Rflag = False
            # 输出结果
            # print("Best mapping (M1 -> M2 index):", best_state)
            # print("Best total similarity:", best_energy)  # 转换为正值
            # 可视化 ES 矩阵和结果(可选)
            # for i in range(n):
                # print(f"M1 {i} -> M2 {best_state[i]}, similarity: {ES[i][best_state[i]]}")
        else:
            # 转置处理
            row_ind, col_ind = linear_sum_assignment(cost_matrix.T)
            best_state = col_ind.tolist()
            best_energy = ES_np[col_ind, row_ind].sum()
            Rflag = True
            # 输出结果
            # print("Best mapping (M2 -> M1 index):", best_state)
            # print("Best total similarity:", best_energy)  # 转换为正值
            # 可视化 ES 矩阵和结果(可选)
            # for i in range(m):
                # print(f"M2 {i} -> M1 {best_state[i]}, similarity: {ES[best_state[i]][i]}")
        return Rflag, best_state, best_energy

    @staticmethod
    def get_model(modelname):
        return EvaluationUtil.MODEL
  
    @staticmethod
    def set_base_model(modelname):
        if modelname in ['all-MiniLM-L6-v2','bge-large-en-v1.5']:
            from sentence_transformers import SentenceTransformer
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            EvaluationUtil.MODEL = SentenceTransformer(os.path.join(EvaluationUtil.MODEL_PATH_BASE,modelname)).to(device)
        if modelname in ['bert-base-uncased','codebert-base']:
            from transformers import AutoTokenizer, AutoModel
            model_path = os.path.join(EvaluationUtil.MODEL_PATH_BASE, modelname)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            EvaluationUtil.__TOKENIZER = tokenizer
            EvaluationUtil.__MODEL = model
    @staticmethod
    def get_base_model(modelname):
        return EvaluationUtil.__TOKENIZER, EvaluationUtil.__MODEL
    @staticmethod
    def get_embedding(text,model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Batch encode all texts
        embeddings = model.encode(text, convert_to_tensor=True, device=device)
        return embeddings
    @staticmethod
    def get_base_embedding(text,tokenizer,model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()} 
        with torch.no_grad():
            outputs = model(**inputs)
        # 使用 [CLS] 向量（BERT 系）或平均池化
        embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        # 或者使用平均池化：embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

    @staticmethod
    def calc_similarity_single(text,text2,modelname): # bert-base-uncased or other models
        if modelname in ['all-MiniLM-L6-v2','bge-large-en-v1.5']:
            model = EvaluationUtil.get_model(modelname)
            embeddings = model.encode([text, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
            return similarity
        elif modelname in ['bert-base-uncased','codebert-base']:
            tokenizer, model = EvaluationUtil.get_base_model(modelname)
            # 计算候选和参考文本的嵌入
            reference_emb = EvaluationUtil.get_base_embedding(text,tokenizer,model)
            candidate_emb = EvaluationUtil.get_base_embedding(text2,tokenizer,model)
            # 计算余弦相似度
            similarity = torch.cosine_similarity(reference_emb, candidate_emb, dim=1).item()
            return similarity
        else: raise ValueError(f"{modelname} is not a correct name! Can't find in hf_models folder")
    @staticmethod
    def calc_similarity_batch(modelname):
        if modelname in ['all-MiniLM-L6-v2','bge-large-en-v1.5']:
            def calc_similarity_SentenceTransformerModel(ES):
                model = EvaluationUtil.get_model(modelname)
                # 计算候选和参考文本的嵌入
                n = len(ES)
                m = len(ES[0])
                pairs = [ES[n_][m_] for n_ in range(n) for m_ in range(m)]
                text, text2 = zip(*pairs)
                reference_emb = EvaluationUtil.get_embedding(text,model)
                candidate_emb = EvaluationUtil.get_embedding(text2,model)
                similarity = torch.cosine_similarity(reference_emb, candidate_emb, dim=1)
                return similarity.view(n, m).cpu().numpy()
            return calc_similarity_SentenceTransformerModel
        elif modelname in ['bert-base-uncased','codebert-base']:
            def calc_similarity_TransformersModel(ES):
                # 计算候选和参考文本的嵌入
                n = len(ES)
                m = len(ES[0])
                pairs = [ES[n_][m_] for n_ in range(n) for m_ in range(m)]
                text, text2 = zip(*pairs)
                tokenizer, model = EvaluationUtil.get_base_model(modelname)
                reference_emb = EvaluationUtil.get_base_embedding(text,tokenizer,model)
                candidate_emb = EvaluationUtil.get_base_embedding(text2,tokenizer,model)
                similarity = torch.cosine_similarity(reference_emb, candidate_emb, dim=1)
                return similarity.view(n, m).cpu().numpy()
            return calc_similarity_TransformersModel
        else: raise ValueError(f"{modelname} is not a correct name! Can't find in hf_models folder")
        