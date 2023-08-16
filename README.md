# QAsystem

# 大致架构

![image](https://github.com/iGangao/QAsystem/assets/73676846/2b086c75-c639-487f-9553-332c36798796)


# 开发流程
## 一、准备数据
### 1、QA数据
#### 数据格式：
![image](https://github.com/iGangao/QAsystem/assets/73676846/88788829-326e-4ead-a848-0c5428327c4d)

### 2. Q-->keyword数据
#### 数据格式：
![image](https://github.com/iGangao/QAsystem/assets/73676846/a9285bd9-816a-4781-9ef4-41ca221863cc)


### 3. keyword matching keyword 数据
#### 数据格式
![image](https://github.com/iGangao/QAsystem/assets/73676846/85b7e4e9-6c26-498c-9623-78825605e351)


## 二、训练keyword extract model
### 1. 使用BERT 预训练模型
#### 主要方法：
- [BERT-KPE](https://github.com/thunlp/BERT-KPE)
- [BERT-Keyword-Extractor](https://github.com/ibatra/BERT-Keyword-Extractor)
- [BERT-keyphrase-extraction](https://github.com/pranav-ust/BERT-keyphrase-extraction)

## 三、构建 vectorDB
### 数据格式
![image](https://github.com/iGangao/QAsystem/assets/73676846/a272e64a-75f4-4966-b723-32b7d4a4b1a3)

### 构建流程
#### 1. keyword to vector
将关键词向量化
#### 2. vcetor store in DB
将向量存入数据库中，并得到该向量的唯一标识（索引）
#### 3. vector index mapping "reference"
将向量索引和数据“reference”映射
	key - value
	key : vector index
	value : reference

## 四、训练keyword matching keyword model
### 1. text2vec
### 2. SimCSE

## 五、LLM finetune
### 1. LORA
#### 数据格式：
	Q:
	A:
### 2....
