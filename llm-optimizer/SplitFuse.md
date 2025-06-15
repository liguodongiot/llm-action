




SplitFuse



调度策略：

0: 先入先出策略（FCFS）。
4：短任务优先（SJF）。
5：长任务优先（LJF）。
6：多级反馈队列（Skip-Join MLFQ）。
7：短任务优先多级反馈队列（SJF-MLFQ）。
建议：一般使用FCFS策略；对于追求极限吞吐的场景建议使用SJF，但是可能会导致长输入等待时间大幅增长；对于希望在吞吐和长输入等待取平衡的使用场景，建议使用Skip-Join MLFQ或SJF-MLFQ策略。

- https://www.hiascend.com/doc_center/source/zh/mindie/10RC3/mindiellm/llmdev/mindie_llm0292.html


