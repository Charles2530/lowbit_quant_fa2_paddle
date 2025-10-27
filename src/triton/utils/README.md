# InferenceKit
应用于大语言模型的评估以及推理

## 1. 项目结构目录
**dataset**: 数据集加载

**models**: LLM模型加载以及不同推理方式的实现

**reward_models**： Reward model的模型加载以及推理方式实现

**utils**: 存储、读取文件的工具类

**data_config.py**：配置评测数据

**model_config.py**：配置模型

**inference.py**: 推理benchmark

**main.py**：推理benchmark的主程序入口

**play.py**：多轮对话的主程序入口

## 运行环境
```
pip install -r requirements.txt
```

## 2. 运行
根据需求的不同运行不同的脚本。
**评测benchmark**：用于批量进行评测
**多轮对话**：通过在命令行输入文本和模型对话

### 评测benchmark
```
cd scripts
bash run.sh
```
**参数介绍：**
work_dir: 结果保存的路径
MODEL_NAME: 评测的模型名字，必须在model_config.py中进行配置。
MODEL_PATH: 评测的模型的权重路径
DATASET_NAME: 评测的benchmark的名字，必须在data_config.py中进行配置。

### 多轮对话
```
cd scripts
bash dialogue.sh
```
**参数介绍：**
model: LLM的模型名称，要出现在model_config.py的配置文件中。
model_path: LLM的模型路径。
reward_model: 奖励模型的名称，同样需要在model_config.py中进行配置
reward_model_path: 奖励模型的本地路径

## 3. 加入模型

### 加入LLM
- step 1 在models中实现相应的LLM的初始化以及推理函数（至少要实现vanilla_generate,一般看官方的调用文档）。
- step 2 在models 的__init__.py中import相关的类。
- step 3 在model_config.py中添加模型的相关配置。在sft_model_groups中加入模型的名称（也就是字典的键，最好简洁明了），值为partial包裹的模型的类名以及模型的本地路径。

### 加入Reward Model
- step 1 在reward_models中实现相应的reward model的初始化函数以及计算得分的函数（score）。一般怎么加载模型和计算score查看官方的调用文档。
- step 2 在models的__init__.py中import 相关的类。
- step 3 在model_config.py中添加模型的相关配置。具体配置同LLM👆


## 4. 加入评测数据集
- step 1 在dataset中实现数据集的加载函数。要求：数据集中必须有的字段：index和input。index标明该样本的唯一标识序号，input代表模型的输入。数据的格式参照example_dataset.json 。 如果需要计算评价指标，请在对应的dataset的类中实现evaluate方法。
- step 2 在dataset 的__init__.py中import相关的类。
- step 3 在data_config.py中配置相关的信息。supported_dataset中的键为数据集的名称（最好简洁明了），value为用partial包裹的对应的数据集的类以及dataset_name（和键一致就行）和dataset_path（数据集的路径）。参见simpledataset.py


## 模型配置
- Config 读取优先级: runtime arguments > command arguments(only generation config) > model arguments > config file > model default > default

## 2024/11/02 Update @ wenbin wang
1. 将wentao修改的代码合并到main分支
2. 增加benchmark 评测的接口函数 evaluate
3. 将benchmark数据迁移到文件夹benchmark中

## 2024/11/07 Updata @ Wentao Jiang
1. 增加Vanilla MCTS代码，在50条数据上测试

## 2024/11/07 Update @ Wenjie Wu & Tong Yu
1. 增加了basedataset的evaluate函数，可以对推理结果进行评测。grader.py中增加了数学表达式评测的函数，包括数学表达式相等评测和数学表达式相似度评测。
2. 修改了 gsm8k 和 math dataset 的prompt， 提示模型将答案放在 `$\boxed{}$` 中，方便评测。

## 2024/11/08 Update @ Tong Yu
1. 增加了compare接口，对比两个生成结果A,B，重点输出A对B错、A错B对、AB均错的情况
2. 针对prompt有box{}的提示词，匹配所有box，只要存在正确回答的就认为正确

## 2024/11/11 Update @ Wenjie Wu & Tong Yu & Wentao Jiang
1. 增加Best of N方法
2. 增加Beam Search方法
3. 重构generation模块代码

## 2024/11/12 Update @ Wenjie Wu & Tong Yu
1. 增加majority_vote, min_vote, last_vote的投票方法
2. 优化了result文件的保存格式，现为dataset_name-cot_method-voting_method.json或dataset_name-cot_method.json


## TODO
1. ~~实时生成实验结果，而不是在进行完所有推理后再生成~~
2. ~~模型配置管理以及实验结果管理：result文件夹中子文件代表数据集，再下级文件夹代表各数据集下进行的实验，需要以model+reward_model+exp_name命名，其中包括config和实验结果~~
3. 将其和仓库understanding-o1/understanding-o1合并
4. ~~增加 chat template，当前的模板不太好~~
5. ~~理顺模型逻辑，重构models.llama3文件，将其改成通用的huggingface model，为每个模型保存配置文件~~
6. ~~分割符作为各个模型的配置，包括model和reward model~~
7. ~~分离inference和evaluate代码，一个用于inference实验，一个用实验生成结果来评估性能~~
8. ~~model和reward_model提供默认路径，支持仅通过名称调用~~
9. ~~增加Best of N, Beam Search的CoT搜索方式~~
10. ~~logger 功能~~
11. ~~Resume 功能~~
12. ~~多Vote~~
13. 模型分片
14. Flash Attention
15. Token-Level搜索框架实现
