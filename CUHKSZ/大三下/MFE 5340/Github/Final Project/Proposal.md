1.  **定义问题**
    - 本项目关注的核心研究问题是：**在中国股票横截面中，若将股票之间的关系结构显式纳入条件潜在因子模型，是否能够比仅依赖个股特征的模型更有效地刻画股票的动态因子暴露（time-varying beta / conditional exposure），并进一步提升样本外定价能力、收益预测能力，以及 long-short 与 long-only 投资组合表现？**
    - Project 重点是把图神经网络放入一个更标准的资产定价框架中：我们真正关心的是 **conditional beta / latent factor learning** 是否因为引入图结构而变得更准确。
	    - Kelly, Pruitt, and Su 的 IPCA 已经表明，个股特征可以作为潜在时变载荷的工具变量；
	    - Gu, Kelly, and Xiu 的 conditional autoencoder 则进一步证明，这种“特征到载荷”的映射可以被非线性化。
	    - 本研究将沿着这条主线继续推进：检验 **graph context 是否能为 dynamic exposure estimation 提供额外信息**。  
    - 这一问题与量化投资高度相关。传统横截面预测模型通常将股票视为彼此独立的样本，但现实市场中，行业联动、风格共振、流动性传导、共同风险暴露和拥挤交易会使个股收益在截面上形成显著的网络依赖。若这些依赖结构确实能改善 beta 或 latent factor 的识别，那么它们就不只是统计上的“附加信息”，而是对投资组合构建有实际价值的定价信息。
    - 这个问题之所以重要，还因为它正好位于课程主题“机器学习与量化投资”的交叉点上：一方面，它保留了 MFE5340 中最核心的资产定价语言——因子、载荷、样本外定价、投资组合构建；另一方面，它也自然吸收了深度学习课程中的 attention / graph representation learning 思想。但在这个项目里，深度学习不是目的本身，而是服务于更严肃的资产定价识别问题。
    - 因此，本项目的研究范畴将被明确限定为：**基于中国股票月频面板数据，在 rolling / expanding 的样本外框架下，构建一个“图结构增强的条件潜在因子定价模型”，并与 IPCA、Conditional Autoencoder 以及不含图结构的 MLP 进行比较。**

2.  **设定目标与研究目的**
    - **目标一：构建一个图结构增强的条件潜在因子定价框架，用于学习中国股票横截面的动态因子暴露。**
        - 研究目的 1.1：定义可实施的股票图结构。
	        - 节点为股票，节点特征为公司特征与必要的市场变量；
	        - 边结构优先采用“**静态行业关系 + 动态相似性关系**”的双层设计，其中动态边可根据过去若干月收益相关性、特征余弦相似度或 kNN 相似网络按月更新。
        - 研究目的 1.2：设计条件暴露函数。
	        - 将股票在时点 $t$ 的暴露写为  
	          $$
	          \beta_{i,t} = g_\theta(X_t, \mathcal{G}_t),
	          $$
	          其中 $X_t$ 为个股特征，$\mathcal{G}_t$ 为股票图，$g_\theta$ 为图注意力或图卷积型网络。
	        - 这里图模型的功能不是直接替代因子模型，而是增强对 **conditional exposure** 的刻画。
        - 研究目的 1.3：在模型目标中加入资产定价导向。
	        - 项目将不只优化 next-month return forecast error，还将尝试加入 latent factor reconstruction loss、cross-sectional pricing error，或对定价误差的惩罚项，使模型在“预测”与“定价”之间取得平衡。
	        - 这样可回应 no-arbitrage / economic target 方向，也使模型更符合资产定价研究逻辑。（在 $g_\theta$ 的输出层或损失函数中尝试嵌入**无套利约束**，通过构建 SDF 并约束其对超额收益的定价性质，确保模型生成的 Beta 在经济学上是合理的，而非纯粹的统计拟合）
        - 研究目的 1.4：控制项目复杂度。
	        - 主规格先使用月频数据、月度更新图和相对简洁的 GAT/GCN 结构；
	        - 更复杂的动态图模块、混频输入或更强的无套利约束可放在扩展分析中，而不作为一开始必须完成的主模型。

    - **目标二：检验图结构带来的增量价值，并评估其经济意义与可解释性。**
        - 研究目的 2.1：设置清晰的 benchmark 体系。至少包括：
            - IPCA：线性、无图的动态暴露模型；
            - Conditional Autoencoder：非线性、无图的动态暴露模型；
            - MLP：不显式建模 latent factor / graph structure 的深度预测基准；
            - Graph model：非线性 + 图上下文的核心模型。
          这一比较将帮助区分：性能提升究竟来自“非线性”，还是来自“关系结构”。
        - 研究目的 2.2：严格开展样本外检验。采用 rolling 或 expanding 窗口，在每个月仅利用历史可得信息训练与调参，再形成下月信号与组合。重点报告 OOS $R^2$、rank IC、预测误差、pricing error，以及不同模型之间的统计与经济差异。
        - 研究目的 2.3：按照课程要求构建两类投资组合：
            - long-short 组合，用于检验模型对横截面排序能力的提升；
            - long-only 组合，用于检验模型在现实投资中的可用性。
          组合评价指标包括年化收益、年化波动率、Sharpe ratio、最大回撤、换手率及稳健性表现。
        - 研究目的 2.4：开展可解释性分析。除常规 permutation importance 外，还将分析 attention 权重、关键邻居节点、重要边类型，以及不同市场状态下关系结构的重要性变化，从而回答模型究竟是通过哪些特征、哪些连接、哪些局部网络来改善动态暴露估计。
        - 研究目的 2.5：开展稳健性分析。包括更换图构建方法、调整股票池（中证500 vs 更大全市场子集）、改变窗口长度、改变因子维度 $K$、以及比较静态图与动态图设定，确保结论不依赖于某个单一设定。

3.  **文献综述**
    - 现有文献大致可以分成四条主线。
    - **第一条主线：传统统计潜因子模型。**  
	    - 传统 PCA 通过协方差结构提取潜在因子，但它本质上只利用二阶矩信息，容易忽略那些方差不大但风险溢价重要的“弱因子”。
	    - Lettau and Pelger 提出的 RP-PCA 在 PCA 目标函数中加入 pricing error 惩罚，说明 latent factor extraction 不应只围绕协方差最大化，而应兼顾资产定价目标。
	    - 这为本项目提供了一个重要启发：如果我们要从图结构中学习 latent factors 或 loadings，那么目标函数也不应只是纯统计意义上的预测误差。  
    - **第二条主线：动态暴露模型。**  
	    - Kelly, Pruitt, and Su 提出的 IPCA 是现代 conditional asset pricing 的关键进展之一。它把公司特征引入 latent loadings 的估计，使因子暴露能够随时间和个体状态变化。
	    - 这个框架非常适合作为本项目的理论起点，因为它明确提出：真正需要学习的是由特征驱动的 **时变 beta**。
	    - 但 IPCA 对特征到载荷的映射仍然是线性的，这限制了其对复杂交互效应的刻画。
    - **第三条主线：非线性 latent factor / deep asset pricing。**  
	    - Gu, Kelly, and Xiu 的 conditional autoencoder 模型将这一映射推广到非线性情形，使公司特征可以通过神经网络影响因子载荷，并显著改善定价误差。
	    - 进一步的深度资产定价研究则表明，把时间变动、宏观状态与无套利条件纳入神经网络框架，可以在样本外定价和投资绩效上取得进一步提升。
	    - 这说明，非线性确实对资产定价问题重要，但也意味着：如果希望提出有说服力的增量贡献，就不能只停留在“比 MLP 更复杂”，而必须回答图结构是否比“无图非线性”还多带来了一层有效信息。
    - **第四条主线：图结构与关系建模。**  
	    - Son and Lee 的 graph-based multi-factor asset pricing model 已经明确把 connectivity 引入 risk exposure estimation，说明图结构不仅可用于一般 stock prediction，也可以直接进入多因子定价框架。
	    - 相关后续研究则进一步强调，若股票之间的关系会随市场状态变化，动态图和 attention 机制可能比静态图更符合金融市场现实。
	    - 与此同时，面向中国市场的近期研究也表明，attention-based 的横截面收益模型在 A 股上具有竞争性的 OOS 表现，说明在中国市场引入更强的交互结构是有经验依据的。
    - 从这些文献中可以看到一个清晰趋势：  
        1. 因子模型正在从静态走向动态；  
        2. 从线性走向非线性；  
        3. 从“只看个股自身特征”走向“同时建模横截面关系结构”；  
        4. 从纯统计降维走向加入经济目标、pricing error 甚至 no-arbitrage 约束的定价框架。
    - 但现有研究仍存在几处空白，正好构成项目的切入点：
        - 第一，很多图模型文献更偏“股票预测”而不是“资产定价”，即它们强调 prediction，却较少把 graph structure 明确嵌入 **conditional beta / latent factor** 的语言中。
        - 第二，已有 conditional beta 模型（如 IPCA、CAE）虽然能处理个股特征导致的时变暴露，但通常没有显式建模股票之间的网络依赖。
        - 第三，中国市场上的相关研究虽已有 attention 或 graph 方法，但真正把“动态图结构 + 条件潜因子定价 + OOS portfolio construction”整合起来的工作仍不多。
    - 因此，本项目的文献定位：  
      **并不是单纯地把 GNN 应用于股票预测，而是尝试在中国股票市场中，把图结构作为对 conditional beta function 的补充信息源，构建一个更贴近资产定价理论的动态图条件潜因子框架，并检验其是否能在 OOS 定价、预测与投资组合构建中提供增量价值。**
    - 在正式 Phase 1 报告中，文献综述应优先围绕以下高质量同行评审论文展开：
        - Kelly, Pruitt, and Su (2019), *Journal of Financial Economics*；
        - Gu, Kelly, and Xiu (2021), *Journal of Econometrics*；
        - Lettau and Pelger (2020), *Journal of Econometrics*；
        - Son and Lee (2022), *Finance Research Letters*；
        - 与经济目标 latent factors、deep asset pricing、以及中国市场 cross-sectional forecasting 直接相关的后续同行评审论文。

