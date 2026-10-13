# Final Project Guideline
*MFE5340: AI in FE: Quantitative Investment*

**Chuan Shi**
School of Data Science
Spring 2026

---

## 1 **Phases and Timeline**

#### Summary
*Table: Phases and Timeline of Final Project*

| Phase | Deadline | Weight | Deliverables and Description |
| :--- | :--- | :---: | :--- |
| 1. Problem Statement and Literature Review | **April 1, 2026 23:59:59** | 20% | Submit a report outlining problem statement and literature review. |
| 2. Design and Development | **April 13 – 17, 2026** | 20% | Each group is required to have a discussion (in-person or online) with me about design progress, initial empirical results, and code to date. This serves as a mid-term project checkpoint. Please make appointments with me. |
| 3. Report and Presentation | Files due date: **May 5, 2026 23:59:59** <br> Final presentation: **May 6 during regular lecture hours** | 60% | Submit the final report, code, and data, and present your findings. This is the final project phase, where all work should be fully implemented and documented. |

#### Grading and Late Submission Policy
- Each phase has a maximum score of 100 points.
- For phase 1, in case of late submissions, 10 points will be deducted for each 24-hour period. For example, a delay of less than 24 hours results in a 10-point deduction, a delay of 24 to less than 48 hours results in a 20-point deduction, and so on.
- For phase 2, it is your group’s responsibility to schedule an in-person discussion with me during the week of 4/13 to 4/17. Failure to do so will result in zero points for phase 2.
- For phase 3, no late submission is allowed.

---

## 2 **Topic: Machine Learning and Quantitative Investment**

#### Goal
- Machine learning techniques offer powerful tools to improve portfolio construction by capturing complex relationships in data, enhancing out-of-sample ( **OOS** ) performance, and addressing limitations of traditional factor models.
- You are provided with individual stock price data and common firm characteristics as potential features.
- Core methods include the Risk-Premium PCA approach by *Lattau and Pelger (2020)* , the general PCA model from *Bryzgalova et al. (2023)* , which incorporates cross-sectional and time-series economic targets, and the autoencoder model by *Gu et al. (2021)* . Linear models like ridge and lasso regression, as well as neural networks discussed in *Gu et al. (2020)* , are viable options.
- You are expected to perform feature selection prior to model training to remove irrelevant features.
- You are encouraged to explore the entire cross-section of the Chinese stock market, but in order to reduce computational load, it is also perfect fine to limit the stock universe to the constituents of the **CSI 500 Index** .
- Transaction costs can be ignored for the purpose of this project. However, you are expected to be aware of practical considerations such as high turnover rates associated with complex machine learning models.

#### Scope
- Select machine learning models to construct portfolios based on the provided stock data and firm characteristics. (It is also okay to use your own data as long as you explain your data source.)
- Perform **OOS** testing using a rolling or expanding window approach. At the end of each month, train and validate the model on historical data to obtain weights for portfolio construction, and evaluate the portfolio’s performance for the next month as an **OOS** test. Rebalance the portfolio monthly.
- To ensure the practical value of this project, please investigate and report **both the long/short hedged portfolio and the long-only portfolios** .
- Explain how you choose your loss function, detailing why the chosen loss function aligns with your project’s goals. Explain your choice of tuning parameters. Outline the validation approach used (e.g., cross-validation, out-of-sample testing) to ensure the model is optimized without overfitting, and describe the steps taken to assess and adjust the model’s performance based on these criteria.
- Analyze the effectiveness of the machine learning approach in enhancing **OOS** performance.
- Emphasize correct model application and interpretability by including feature importance analysis, such as permutation importance, to understand the contribution of each feature to model predictions.
- Summarize findings, focusing on the practical application of machine learning in quantitative investment and insights gained from feature importance or other interpretability metrics.

#### Group of 1 vs. Group of 2: Scope
- **1-Person Groups:**
    - Shall use (at least) one machine learning model (can be linear or non-linear).
    - Required to analyze feature importance.
- **2-Person Groups:**
    - Must use (at least) two different machine learning models (one model per person; can be linear or non-linear).
    - Required to analyze feature importance for each of the models and compare the differences in feature importance across the models.
- For other aspects of the project scope covered, the requirements are the same for both 1-person and 2-person groups.

#### Group of 1 vs. Group of 2: Deliverables
- Phase 1 report should be submitted as a group. There is no need to specify the contributions of individual members.
- The final report must also be submitted as a group. However, each member is required to take responsibility for writing a specific section of the report (e.g., data processing, model design, results analysis, etc.). Each section must clearly indicate the author.
- Despite the division of responsibilities, the project aims to encourage thorough discussion and collaboration within the group. Therefore, while individuals are responsible for specific sections, the entire group is collectively accountable for the report as a whole.
- During the final presentation, each group member must present their assigned section individually and answer related questions.

---

## 3 **Phase 1 Guideline**

#### Objective
- This phase aims to establish a clear and well-defined research problem while providing a literature review that situates the project within the existing body of knowledge.
- You are expected to develop a foundational understanding of the problem, its significance, and relevant prior research.

#### Guideline for Problem Statement
1.  **Define the Problem**
    - Clearly articulate the research problem, describing its relevance to quantitative investment.
    - Explain why the problem is important, particularly in the context of applying machine learning to backtesting strategies.
    - Provide sufficient background to help understand the problem and its significance within the scope of the course.
2.  **Set Goals and Objectives**
    - Identify one or two main goals for the project, directly linked to addressing the problem.
    - Break down each goal into specific, measurable objectives, detailing the planned steps to achieve the project’s aims (e.g., methodological approach, data analysis, or model building).
3.  **Literature Review**
    - Summarize key studies and contributions in the field related to the problem.
    - Discuss trends, shared findings, or divergent results among relevant research, demonstrating a clear understanding of the topic’s background.
    - Identify any remaining gaps or unresolved issues in the literature that your project will address.
    - **Important:** The literature review must be based on high-quality, peer-reviewed journal papers. Web reports, Wikipedia, or non-peer-reviewed sources are **NOT** acceptable.

#### Deliverables and Deadline
- **Deliverables:** A report with:
    - A clearly defined problem statement.
    - Articulated goals and objectives.
    - A literature review that demonstrates an understanding of key sources, current trends, and existing gaps.
- **Deadline:** April 1, 2026, 23:59:59 (This is a **Wednesday** .)
- **Word Count Requirement:** Minimum 1,000 words.

#### Grading Rubric for Phase 1

| Criteria | Points |
| :--- | :--- |
| Clarity and Relevance of Problem Statement | 30 Points |
| Relevance and Structure of Goals/Objectives | 20 Points |
| Depth and Insight of Literature Review | 30 Points |
| Writing Quality | 10 Points |
| Source Quality and Citation | 10 Points |

##### Clarity and Relevance of Problem Statement (30 Points)

- **26–30 Points:** The problem is clearly and specifically articulated, with strong relevance to quantitative investment and machine learning backtesting strategies. The problem’s importance is well-explained, supported by sufficient background that makes it understandable and significant.
- **21–25 Points:** The problem is clearly stated but may lack some specificity or depth, and its relevance to quantitative investment is not strongly emphasized. The importance of the problem is explained, but the background provided is somewhat limited or general.
- **16–20 Points:** The problem is somewhat clear but may be vague, overly broad, or loosely connected to quantitative investment. The background is limited or superficial, making the problem’s significance less clear.
- **0–15 Points:** The problem is unclear, vague, or irrelevant to quantitative investment, with little to no connection to machine learning. Little to no background is provided, leaving the problem’s importance unestablished.

##### Relevance and Structure of Goals/Objectives (20 Points)

- **18–20 Points:** Goals are directly tied to the problem statement and are highly relevant. Objectives are specific, measurable, actionable, and logically sequenced. The relationship between goals and objectives is clearly explained, showing a clear plan for addressing the problem.
- **15–17 Points:** Goals are relevant and connected to the problem statement but may lack some specificity or depth. Objectives are clear and actionable but may not be fully measurable or logically sequenced.
- **11–14 Points:** Goals are somewhat relevant but may be vague or generic. Objectives are present but lack clarity, specificity, or logical structure.
- **0–10 Points:** Goals are irrelevant, poorly defined, or missing. Objectives are absent or unclear, with no actionable steps provided.

##### Depth and Insight of Literature Review (30 Points)

- **26–30 Points:** Comprehensive review of high-quality, peer-reviewed studies directly related to the problem. Demonstrates a clear understanding of key concepts and findings in the field. Identifies practical challenges in the application of machine learning to backtesting strategies.
- **21–25 Points:** Covers relevant studies with some critical analysis, but the review may lack depth or breadth. Demonstrates a good understanding of the topic but may miss some key challenges.
- **16–20 Points:** Review includes relevant studies but lacks critical analysis or depth. Shows a basic understanding of the topic but fails to highlight challenges clearly.
- **0–15 Points:** Literature review is incomplete, superficial, or relies on low-quality or irrelevant sources. Lacks critical analysis or understanding of the topic.

##### Writing Quality (10 Points)

- **9–10 Points:** Well-structured and clearly written, with excellent grammar, spelling, and logical flow. Ideas are presented in a concise and coherent manner.
- **7–8 Points:** Generally well-written but may have minor issues with grammar, structure, or flow.
- **5–6 Points:** Writing is understandable but contains noticeable issues with grammar, structure, or clarity.
- **0–4 Points:** Writing is unclear, poorly structured, or riddled with grammatical errors.

##### Source Quality and Citation (10 Points)

- **9–10 Points:** All sources are high-quality, peer-reviewed journal articles. Citations are consistent, accurate, and follow the required format (e.g., APA, Chicago).
- **7–8 Points:** Sources are mostly high-quality, with minor issues in citation formatting or consistency.
- **5–6 Points:** Some sources are low-quality or non-peer-reviewed, and citation formatting is inconsistent.
- **0–4 Points:** Sources are of poor quality or missing, and citations are incorrect or absent.
