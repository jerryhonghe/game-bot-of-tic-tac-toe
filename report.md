# Chapter 1: Introduction
# 第一章：引言

## 1.1 Project Background and Motivation
## 1.1 项目背景与动机

Board games have long served as ideal testing grounds for artificial intelligence research. Their well-defined rules, discrete state spaces, and clear winning conditions create contained environments where algorithmic approaches can be evaluated systematically. Since Claude Shannon's pioneering work on chess-playing machines in the 1950s and IBM Deep Blue's victory over world champion Garry Kasparov in 1997, board games have consistently marked milestones in AI development. More recently, DeepMind's AlphaGo and AlphaZero have demonstrated extraordinary capabilities in complex games like Go and chess, revolutionizing our understanding of what machine learning systems can achieve.

棋类游戏长期以来一直是人工智能研究的理想测试平台。它们具有明确定义的规则、离散的状态空间和清晰的获胜条件，为算法方法的系统评估创造了封闭的环境。从20世纪50年代克劳德·香农关于国际象棋机器的开创性工作，到1997年IBM深蓝战胜世界冠军加里·卡斯帕罗夫，棋类游戏一直标志着人工智能发展的里程碑。近年来，DeepMind的AlphaGo和AlphaZero在围棋和国际象棋等复杂游戏中展示了非凡的能力，彻底改变了我们对机器学习系统可以实现的认识。

Tic Tac Toe and Connect Four represent particularly interesting subjects for AI research. Tic Tac Toe, with its simple 3×3 grid, offers a tractable environment where the entire game tree can be exhaustively explored. Despite its apparent simplicity, it provides a perfect starting point for understanding fundamental concepts in game-playing AI, including minimax search, state evaluation, and reinforcement learning principles. Connect Four, played on a 6×6 grid in our implementation (compared to the traditional 7×6), increases complexity significantly. Its larger state space introduces practical computational limitations that mirror challenges faced in more sophisticated domains, making it an excellent stepping stone between trivial games and truly complex ones like chess or Go.

井字棋和四子棋是人工智能研究中特别有趣的研究对象。井字棋采用简单的3×3网格，提供了一个可以穷尽探索整个博弈树的环境。尽管表面上看起来简单，但它为理解博弈AI中的基本概念提供了完美的起点，包括极小极大搜索、状态评估和强化学习原理。在我们的实现中，四子棋在6×6网格上进行（相比传统的7×6），显著增加了复杂性。它更大的状态空间引入了实际的计算限制，反映了在更复杂领域面临的挑战，使其成为简单游戏和真正复杂游戏（如国际象棋或围棋）之间的绝佳过渡。

The value of comparing different AI approaches within these environments extends beyond mere academic interest. By implementing and analyzing Q-learning (reinforcement learning), Minimax (deterministic search), and Monte Carlo Tree Search with neural networks (inspired by AlphaZero), we create a practical framework for understanding the relative strengths and limitations of these algorithmic paradigms. These methods represent fundamentally different approaches to problem-solving: Minimax embodies perfect information exhaustive search, Q-learning exemplifies learning through trial and error without explicit modeling, and AlphaZero-style approaches combine neural networks for evaluation with tree search for planning.

在这些环境中比较不同人工智能方法的价值超出了纯学术兴趣。通过实现和分析Q-learning（强化学习）、Minimax（确定性搜索）和结合神经网络的蒙特卡洛树搜索（受AlphaZero启发），我们创建了一个实用框架，用于理解这些算法范式的相对优势和局限性。这些方法代表了根本不同的问题解决方法：Minimax体现了完全信息的穷举搜索，Q-learning通过反复试验而不需要显式建模来学习，而AlphaZero风格的方法则将神经网络用于评估与树搜索用于规划相结合。

My personal motivation for this project stems from a fascination with how different AI approaches can yield similar performance through entirely different mechanisms. Having studied these algorithms theoretically, I wanted to create a platform where their performances could be directly compared, their decision-making processes visualized, and their relative advantages clearly demonstrated. This project also aligns with my interest in educational tools that make AI concepts more accessible and intuitive through interactive demonstration.

我对这个项目的个人兴趣源于对不同AI方法如何通过完全不同的机制产生相似性能的着迷。在理论上学习了这些算法后，我希望创建一个平台，可以直接比较它们的性能，可视化它们的决策过程，并清晰地展示它们的相对优势。这个项目也符合我对教育工具的兴趣，这些工具通过交互式演示使AI概念更容易理解和直观。

## 1.2 Research Objectives and Scope
## 1.2 研究目标与范围

The primary objective of this project is to develop a comprehensive comparison framework for evaluating different AI approaches in board games, with specific focus on Q-learning, Minimax with alpha-beta pruning, and deep learning combined with Monte Carlo Tree Search. Through this comparison, the research aims to illuminate the strengths, weaknesses, computational requirements, and scaling properties of each approach.

本项目的主要目标是开发一个全面的比较框架，用于评估棋类游戏中不同的AI方法，特别关注Q-learning、带alpha-beta剪枝的Minimax以及结合蒙特卡洛树搜索的深度学习。通过这种比较，研究旨在阐明每种方法的优势、劣势、计算需求和扩展特性。

Specific research questions addressed include:

具体研究问题包括：

1. How does the performance of heuristic search methods like Minimax compare to learning-based approaches like Q-learning in environments of varying complexity?
2. What are the key efficiency tradeoffs between these algorithms in terms of computational resources, training requirements, and decision quality?
3. How effectively can reinforcement learning approaches like Q-learning develop optimal or near-optimal strategies through self-play without explicit domain knowledge?
4. To what extent does the integration of neural networks with tree search methods overcome the limitations of traditional approaches in more complex games?
5. How can the decision-making processes of these different algorithms be effectively visualized and understood?

1. 在不同复杂度的环境中，像Minimax这样的启发式搜索方法与像Q-learning这样的基于学习的方法相比，性能如何？
2. 这些算法在计算资源、训练需求和决策质量方面有哪些关键的效率权衡？
3. 像Q-learning这样的强化学习方法在没有明确领域知识的情况下，通过自我对弈能多有效地开发出最优或接近最优的策略？
4. 神经网络与树搜索方法的集成在多大程度上克服了传统方法在更复杂游戏中的局限性？
5. 如何有效地可视化和理解这些不同算法的决策过程？

The scope of this project is deliberately constrained to two games: Tic Tac Toe (3×3 grid) and Connect Four (implemented on a 6×6 grid rather than the standard 7×6). This scope allows for meaningful algorithm comparison while remaining computationally tractable for development and testing on standard hardware. For Connect Four, a smaller 6×6 grid was chosen to reduce the state space while preserving the game's essential strategic elements.

本项目的范围有意限制在两个游戏上：井字棋（3×3网格）和四子棋（在6×6网格上实现，而非标准的7×6）。这个范围允许在保持在标准硬件上开发和测试的计算可行性的同时进行有意义的算法比较。对于四子棋，选择了较小的6×6网格以减少状态空间，同时保留游戏的基本战略元素。

The implementation focuses on three primary AI approaches:

实现主要关注三种AI方法：

1. Q-learning: A model-free reinforcement learning algorithm that learns values for state-action pairs through trial and error.
2. Minimax with alpha-beta pruning: A classical game tree search algorithm that evaluates positions by considering all possible future states.
3. A simplified AlphaZero-style approach: Combining convolutional neural networks with Monte Carlo Tree Search to evaluate positions and plan moves.

1. Q-learning：一种无模型的强化学习算法，通过试错学习状态-动作对的价值。
2. 带alpha-beta剪枝的Minimax：一种经典的博弈树搜索算法，通过考虑所有可能的未来状态来评估位置。
3. 简化的AlphaZero风格方法：结合卷积神经网络与蒙特卡洛树搜索来评估位置和规划移动。

While more sophisticated algorithms and implementations exist, these three approaches represent distinct paradigms in AI research and provide sufficient contrast for meaningful analysis. The project explicitly excludes more advanced variants (such as deep Q-networks or transformers) to maintain focus and feasibility.

虽然存在更复杂的算法和实现，但这三种方法代表了AI研究中不同的范式，为有意义的分析提供了足够的对比。该项目明确排除了更高级的变体（如深度Q网络或transformers），以保持焦点和可行性。

## 1.3 Literature Review
## 1.3 相关文献综述

### 1.3.1 Evolution of AI Algorithms in Board Games
### 1.3.1 棋类游戏中AI算法的演变

The history of AI in board games stretches back to the mid-20th century. Samuel's checkers program (1959) demonstrated how computers could learn to play games through self-improvement, establishing early principles of reinforcement learning. The development of the Minimax algorithm and its optimization through alpha-beta pruning (Knuth and Moore, 1975) provided a foundation for game tree search that remained dominant for decades. Chess programs like Deep Blue (Campbell et al., 2002) relied primarily on these search techniques combined with handcrafted evaluation functions to achieve grandmaster-level play.

棋类游戏中的AI历史可以追溯到20世纪中叶。Samuel的跳棋程序（1959）展示了计算机如何通过自我改进学习玩游戏，建立了强化学习的早期原则。Minimax算法的发展及其通过alpha-beta剪枝的优化（Knuth和Moore，1975）为博弈树搜索提供了基础，这一基础在几十年内一直占据主导地位。像Deep Blue（Campbell等，2002）这样的国际象棋程序主要依靠这些搜索技术结合手工制作的评估函数来实现大师级别的对弈。

In simpler games like Tic Tac Toe, exhaustive search techniques like Minimax can identify the perfect strategy, as demonstrated by Michie (1963) who implemented MENACE (Matchbox Educable Noughts And Crosses Engine), an early reinforcement learning system. For Connect Four, Allis (1988) proved that the first player can force a win with perfect play, using a combination of database construction and search techniques. These early works demonstrated both the power of exhaustive search in tractable games and its limitations in more complex environments.

在像井字棋这样的简单游戏中，Minimax等穷举搜索技术可以识别完美策略，正如Michie（1963）所展示的那样，他实现了MENACE（Matchbox Educable Noughts And Crosses Engine），这是一个早期的强化学习系统。对于四子棋，Allis（1988）证明了先手玩家在完美对弈的情况下可以强制获胜，使用了数据库构建和搜索技术的组合。这些早期工作既展示了穷举搜索在可处理游戏中的威力，也展示了它在更复杂环境中的局限性。

### 1.3.2 Reinforcement Learning in Game AI
### 1.3.2 博弈AI中的强化学习

Reinforcement learning represents a fundamentally different approach to game AI, focusing on learning optimal policies through experience rather than explicit search. Tesauro's TD-Gammon (1995) demonstrated that temporal difference learning could achieve expert-level backgammon play without extensive domain knowledge. Similarly, Q-learning (Watkins and Dayan, 1992) provided a model-free approach to learning action values, making it particularly suitable for environments where constructing accurate models is difficult.

强化学习代表了博弈AI的一种根本不同的方法，专注于通过经验而非显式搜索来学习最优策略。Tesauro的TD-Gammon（1995）证明了时间差分学习可以在没有广泛领域知识的情况下实现专家级别的西洋双陆棋游戏。同样，Q-learning（Watkins和Dayan，1992）提供了一种无模型的方法来学习行动价值，使其特别适合于难以构建准确模型的环境。

For Tic Tac Toe specifically, Sutton and Barto (2018) use the game as an illustrative example for reinforcement learning concepts in their seminal textbook, demonstrating how value functions can be learned through self-play. Mendes et al. (2016) extended reinforcement learning approaches to Connect Four, comparing Q-learning with more sophisticated algorithms and highlighting the challenges of handling larger state spaces. Their work demonstrated that while Q-learning can achieve strong play in modestly complex environments, careful state representation and exploration strategies are crucial.

特别是对于井字棋，Sutton和Barto（2018）在他们的开创性教科书中使用这个游戏作为强化学习概念的说明性例子，展示了如何通过自我对弈学习价值函数。Mendes等人（2016）将强化学习方法扩展到四子棋，比较了Q-learning与更复杂的算法，并强调了处理更大状态空间的挑战。他们的工作表明，虽然Q-learning可以在适度复杂的环境中实现强大的对弈，但谨慎的状态表示和探索策略至关重要。

As van Otterlo and Wiering (2012) note in their comprehensive survey, reinforcement learning in games faces fundamental challenges of exploration-exploitation balance, credit assignment over long sequences, and the curse of dimensionality as games increase in complexity. These limitations have motivated hybrid approaches that combine learning with search.

正如van Otterlo和Wiering（2012）在他们全面的调查中指出的那样，游戏中的强化学习面临着探索-利用平衡的基本挑战、在长序列上的信用分配以及随着游戏复杂性增加而产生的维度灾难。这些限制促使人们采用将学习与搜索相结合的混合方法。

### 1.3.3 Deep Learning and Monte Carlo Tree Search Integration
### 1.3.3 深度学习与蒙特卡洛树搜索的结合

The integration of deep learning with Monte Carlo Tree Search (MCTS) represents one of the most significant advances in game AI of the past decade. While MCTS alone had demonstrated success in games like Go (Coulom, 2006; Gelly et al., 2012), it was the combination with deep neural networks in AlphaGo (Silver et al., 2016) that achieved superhuman performance. The subsequent development of AlphaZero (Silver et al., 2018) generalized this approach across multiple games without game-specific knowledge beyond the rules.

深度学习与蒙特卡洛树搜索（MCTS）的结合代表了过去十年中博弈AI最重要的进步之一。虽然单独的MCTS在围棋等游戏中已经取得了成功（Coulom，2006；Gelly等，2012），但正是在AlphaGo（Silver等，2016）中与深度神经网络的结合才实现了超人的表现。随后开发的AlphaZero（Silver等，2018）将这种方法泛化到多种游戏，除了规则之外不需要特定于游戏的知识。

As Anthony et al. (2017) explain, the neural networks in these systems serve dual purposes: a policy network guides search toward promising moves, while a value network provides position evaluations without requiring full rollouts. This synergy addresses limitations of both neural networks (which struggle with long-term planning) and pure tree search methods (which face combinatorial explosion in complex games).

正如Anthony等人（2017）所解释的，这些系统中的神经网络有双重目的：策略网络引导搜索朝向有希望的行动，而价值网络提供位置评估而无需完整展开。这种协同作用解决了神经网络（在长期规划方面存在困难）和纯树搜索方法（在复杂游戏中面临组合爆炸）的限制。

For simpler games like Connect Four, adaptations of these approaches have been explored by researchers such as Kenny et al. (2020), who implemented a simplified version of AlphaZero for Connect Four and similar games. Their work demonstrated that even with modest computational resources, neural-guided MCTS could outperform traditional approaches. Similarly, Zhao et al. (2019) examined the effectiveness of transfer learning in these contexts, showing how neural networks trained on simplified variants could accelerate learning on more complex versions of games.

对于像四子棋这样的简单游戏，研究人员如Kenny等人（2020）探索了这些方法的改编，他们为四子棋和类似游戏实现了AlphaZero的简化版本。他们的工作表明，即使使用适度的计算资源，神经引导的MCTS也可以胜过传统方法。同样，Zhao等人（2019）研究了迁移学习在这些环境中的有效性，展示了如何利用在简化变体上训练的神经网络加速在更复杂的游戏版本上的学习。

### 1.3.4 Limitations of Current Approaches and Research Gaps
### 1.3.4 当前方法的局限性与研究空白

Despite significant advances, current AI approaches to board games face several limitations. As Wang et al. (2021) observe, reinforcement learning methods struggle with credit assignment and sample efficiency, often requiring millions of training games even for relatively simple environments. Traditional search methods, while effective in bounded contexts, face exponential complexity growth and rely heavily on domain-specific heuristics. Even AlphaZero-style approaches require substantial computational resources for training and may converge to suboptimal strategies in certain contexts (McAleer et al., 2020).

尽管取得了重大进展，但当前的棋类游戏AI方法仍面临几个限制。正如Wang等人（2021）所观察到的，强化学习方法在信用分配和样本效率方面存在困难，即使对于相对简单的环境也常常需要数百万次训练游戏。传统的搜索方法虽然在有限的环境中有效，但面临指数级的复杂性增长，并且严重依赖于特定领域的启发式方法。即使是AlphaZero风格的方法也需要大量的计算资源进行训练，并且在某些情况下可能会收敛到次优策略（McAleer等人，2020）。

A notable gap in current research is the lack of comprehensive, accessible frameworks for direct comparison of these different AI paradigms in controlled environments. While individual algorithms have been extensively studied, comparative analyses that examine performance, interpretability, and scalability across multiple games and approaches remain limited. Additionally, there is a need for more intuitive visualization of AI decision-making processes to enhance understanding of how these algorithms operate in practice.

当前研究中一个显著的空白是缺乏全面、易于使用的框架，用于在受控环境中直接比较这些不同的AI范式。虽然单个算法已经被广泛研究，但检查多种游戏和方法的性能、可解释性和可扩展性的比较分析仍然有限。此外，还需要更直观地可视化AI决策过程，以增强对这些算法在实践中如何运作的理解。

This project aims to address these gaps by creating a platform that not only implements these diverse algorithms but also provides tools for their systematic comparison and visualization. By focusing on games of varying complexity but maintaining a unified experimental framework, it offers a unique opportunity to understand the comparative advantages of different AI approaches across the spectrum of complexity.

该项目旨在通过创建一个平台来解决这些差距，该平台不仅实现这些多样化的算法，还提供工具进行系统比较和可视化。通过专注于不同复杂性的游戏但保持统一的实验框架，它提供了一个独特的机会来理解不同AI方法在复杂性谱系上的比较优势。

## 1.4 Overview of Key Design Decisions and Implementation Challenges
## 1.4 主要设计决策与实现挑战概述

### 1.4.1 Key Design Decisions
### 1.4.1 关键设计决策

The development of this comparative AI platform necessitated several critical design decisions that shaped the project's architecture and functionality. First, a web-based implementation was chosen to maximize accessibility and enable intuitive visualization. The system employs a client-server architecture, with the frontend developed in JavaScript to handle the game interface and visualization, while the backend uses Python to implement the more computationally intensive AI algorithms.

这个比较AI平台的开发需要几个关键的设计决策，这些决策塑造了项目的架构和功能。首先，选择了基于Web的实现，以最大限度地提高可访问性并实现直观的可视化。系统采用客户端-服务器架构，前端使用JavaScript开发，处理游戏界面和可视化，而后端使用Python实现更计算密集型的AI算法。

For algorithm design, the Q-learning implementation incorporates a tabular approach with state abstraction techniques to manage the state space effectively. The Minimax algorithm employs alpha-beta pruning and transposition tables to optimize search efficiency. The AlphaZero-inspired implementation uses a convolutional neural network architecture tailored to board representation, combined with a simplified MCTS implementation to balance performance with computational feasibility.

对于算法设计，Q-learning实现采用了表格方法和状态抽象技术，以有效管理状态空间。Minimax算法使用alpha-beta剪枝和置换表来优化搜索效率。受AlphaZero启发的实现使用了专为棋盘表示定制的卷积神经网络架构，结合简化的MCTS实现，平衡了性能与计算可行性。

A critical design choice was the inclusion of extensive logging and visualization capabilities. The platform records detailed information about each algorithm's decision-making process, including evaluated positions, exploration paths, and confidence values. This information is then rendered through an interactive interface that allows users to step through games and examine AI reasoning at each decision point.

一个关键的设计选择是包含广泛的日志记录和可视化功能。该平台记录了有关每个算法决策过程的详细信息，包括评估的位置、探索路径和置信值。然后，这些信息通过交互式界面呈现，允许用户逐步浏览游戏并检查每个决策点的AI推理。

### 1.4.2 Implementation Challenges
### 1.4.2 实现挑战

Several significant implementation challenges arose during development. The first was managing computational efficiency, particularly for the Connect Four environment. The AlphaZero approach required careful optimization to run effectively without specialized hardware, necessitating compromises in network architecture and MCTS simulation count. Similarly, the Q-learning implementation required effective state representation strategies to manage the larger state space of Connect Four without memory exhaustion.

在开发过程中出现了几个重大的实现挑战。首先是管理计算效率，特别是对于四子棋环境。AlphaZero方法需要仔细优化才能在没有专用硬件的情况下有效运行，这需要在网络架构和MCTS模拟计数方面做出妥协。同样，Q-learning实现需要有效的状态表示策略，以管理四子棋更大的状态空间而不耗尽内存。

Another challenge was developing a common evaluation framework that could fairly compare fundamentally different algorithms. This required careful consideration of metrics beyond simple win rates, including decision time, position evaluation accuracy, and exploration efficiency. Creating meaningful visualizations of these complex decision processes proved particularly challenging, requiring creative solutions to represent high-dimensional data in an intuitive format.

另一个挑战是开发一个公共评估框架，可以公平地比较根本不同的算法。这需要仔细考虑超越简单胜率的指标，包括决策时间、位置评估准确性和探索效率。创建这些复杂决策过程的有意义的可视化证明特别具有挑战性，需要创造性的解决方案以直观的格式表示高维数据。

Integration between the frontend and backend components also presented technical difficulties, particularly in synchronizing game state and efficiently communicating AI decisions with their associated explanatory data. This required careful API design and optimization of data transfer formats to maintain responsive performance.

前端和后端组件之间的集成也带来了技术困难，特别是在同步游戏状态和有效沟通AI决策及其相关解释数据方面。这需要仔细的API设计和数据传输格式的优化，以保持响应性能。

## 1.5 Thesis Structure Overview
## 1.5 论文结构概述

The remainder of this thesis is organized as follows:

本论文的其余部分按如下方式组织：

Chapter 2 provides the theoretical foundations necessary for understanding the implemented algorithms. It explores the principles of game theory, explains the mechanics of Minimax search with alpha-beta pruning, introduces reinforcement learning concepts with focus on Q-learning, and covers the neural network and Monte Carlo Tree Search integration that underpins AlphaZero-style approaches.

第二章提供理解所实现算法所需的理论基础。它探讨了博弈论原理，解释了带alpha-beta剪枝的Minimax搜索机制，介绍了强化学习概念，重点是Q-learning，并涵盖了神经网络和蒙特卡洛树搜索集成，这是AlphaZero风格方法的基础。

Chapter 3 details the system design and methodology, describing the overall architecture, game representations, algorithm implementations, and the evaluation framework used for comparative analysis. It also explains the research methodology employed throughout the project.

第三章详细介绍了系统设计和方法论，描述了总体架构、游戏表示、算法实现以及用于比较分析的评估框架。它还解释了整个项目中使用的研究方法。

Chapter 4 focuses on implementation details, covering the specific technical aspects of the frontend interface, backend services, and the implementation challenges encountered for each algorithm. This chapter provides insights into the practical considerations involved in bringing theoretical algorithms to functional reality.

第四章专注于实现细节，涵盖前端界面、后端服务的具体技术方面，以及每种算法遇到的实现挑战。本章提供了将理论算法应用到功能现实中所涉及的实际考虑因素的见解。

Chapter 5 presents the experimental results and analysis, comparing algorithm performance across various metrics including win rates, decision quality, computational efficiency, and learning capability. It includes visualizations of decision processes and discusses the strengths and weaknesses revealed through systematic testing.

第五章呈现了实验结果和分析，比较了各种指标的算法性能，包括胜率、决策质量、计算效率和学习能力。它包括决策过程的可视化，并讨论了通过系统测试揭示的优势和弱点。

Chapter 6 concludes the thesis with a critical discussion of the project's achievements and limitations. It reflects on key findings, discusses the broader implications for AI research and education, and suggests directions for future work that could extend or improve upon this research.

第六章以对项目成就和局限性的批判性讨论结束论文。它反思了关键发现，讨论了对AI研究和教育的更广泛影响，并提出了可以扩展或改进这项研究的未来工作方向。

The appendices provide supplementary material including the user manual, maintenance documentation, and additional technical details that support but are not central to the main research narrative.

附录提供了补充材料，包括用户手册、维护文档以及支持但不是主要研究叙述核心的其他技术细节。
# Chapter 2: Theoretical Foundations
# 第二章：理论基础

## 2.1 Game Theory Fundamentals
## 2.1 博弈论基础

Game theory provides the mathematical framework for analyzing strategic interactions between rational decision-makers. Board games like Tic Tac Toe and Connect Four represent perfect information zero-sum games, a category particularly amenable to algorithmic analysis. In a perfect information game, all players have complete knowledge of all previous moves and the current state. In zero-sum games, players' interests are directly opposed – one player's gain exactly equals the other player's loss.

博弈论为分析理性决策者之间的战略互动提供了数学框架。井字棋和四子棋等棋类游戏代表了完全信息零和博弈，这是一类特别适合算法分析的博弈类型。在完全信息博弈中，所有玩家都完全了解所有先前的移动和当前状态。在零和博弈中，玩家的利益直接相反——一个玩家的收益恰好等于另一个玩家的损失。

Formally, we can model these board games as a tuple $(S, A, T, R, \gamma)$ where:
- $S$ is the set of all possible board states
- $A$ is the set of all possible actions (moves)
- $T: S \times A \rightarrow S$ is the transition function determining the next state
- $R: S \times A \times S \rightarrow \mathbb{R}$ is the reward function
- $\gamma \in [0,1]$ is a discount factor weighting future rewards

从形式上看，我们可以将这些棋类游戏建模为一个元组 $(S, A, T, R, \gamma)$，其中：
- $S$ 是所有可能的棋盘状态集合
- $A$ 是所有可能的动作（移动）集合
- $T: S \times A \rightarrow S$ 是确定下一个状态的转移函数
- $R: S \times A \times S \rightarrow \mathbb{R}$ 是奖励函数
- $\gamma \in [0,1]$ 是对未来奖励进行加权的折扣因子

The complexity of these games can be quantified through their game tree size. For Tic Tac Toe, the state space contains 3^9 = 19,683 possible board configurations, but many are unreachable in valid play. Accounting for symmetries and gameplay constraints, there are approximately 765 distinct game states. By contrast, Connect Four on a standard 7×6 board has approximately 4.5 trillion reachable states, making exhaustive analysis computationally infeasible. Our implementation uses a 6×6 board, reducing complexity while preserving strategic depth.

这些游戏的复杂性可以通过其游戏树大小来量化。对于井字棋，状态空间包含 3^9 = 19,683 种可能的棋盘配置，但在有效对弈中许多是不可达的。考虑对称性和游戏规则约束，大约有 765 个不同的游戏状态。相比之下，标准 7×6 棋盘上的四子棋大约有 4.5 万亿个可达状态，使穷举分析在计算上不可行。我们的实现使用 6×6 棋盘，减少了复杂性同时保留了战略深度。

The concept of Nash equilibrium is central to game theory – a state where no player can improve their outcome by unilaterally changing their strategy. In deterministic games like Tic Tac Toe and Connect Four, with optimal play, there exists a deterministic Nash equilibrium. Tic Tac Toe, when played optimally, always results in a draw, making it a "solved game." Connect Four is also theoretically solved – with optimal play, the first player can force a win, as proven by Victor Allis in 1988 using a combination of threat-space search and proof-number search.

纳什均衡的概念是博弈论的核心——一种状态，其中没有玩家可以通过单方面改变他们的策略来改善结果。在像井字棋和四子棋这样的确定性博弈中，在最优对弈下，存在确定性纳什均衡。井字棋在最优对弈下总是导致平局，使其成为"已解决的游戏"。四子棋在理论上也是已解决的——在最优对弈下，先手玩家可以强制获胜，这一点由Victor Allis在1988年使用威胁空间搜索和证明数搜索的组合证明。

Game trees represent all possible game sequences, with nodes representing states and edges representing moves. The branching factor – the average number of legal moves from a position – significantly impacts computational complexity. Tic Tac Toe has a maximum branching factor of 9 (first move), decreasing as the board fills. Connect Four's branching factor starts at 6 (number of columns on our reduced board) but remains relatively constant throughout the game, contributing to its greater complexity.

博弈树表示所有可能的博弈序列，节点表示状态，边表示移动。分支因子——一个位置的平均合法移动数——显著影响计算复杂性。井字棋的最大分支因子为9（第一步），随着棋盘填充而减少。四子棋的分支因子从6开始（我们简化棋盘上的列数），但在整个游戏中保持相对恒定，这导致了其更大的复杂性。

## 2.2 Minimax Algorithm and Alpha-Beta Pruning
## 2.2 极小极大算法与Alpha-Beta剪枝

### 2.2.1 Algorithm Principles
### 2.2.1 算法原理

The Minimax algorithm is a deterministic decision rule for determining optimal play in two-player zero-sum games. It recursively evaluates the game tree, assuming that both players make optimal moves. At each decision point, the algorithm alternates between maximizing the player's score and minimizing the opponent's score, hence the name "minimax."

极小极大算法是一种确定性决策规则，用于确定两人零和博弈中的最优对弈。它递归地评估博弈树，假设双方玩家都采取最优移动。在每个决策点，算法在最大化玩家得分和最小化对手得分之间交替，因此得名"极小极大"。

Formally, the value of a state $s$ for the maximizing player is defined recursively:

$$V(s) = \begin{cases} 
U(s) & \text{if } s \text{ is terminal} \\
\max_{a \in A(s)} V(T(s,a)) & \text{if player's turn} \\
\min_{a \in A(s)} V(T(s,a)) & \text{if opponent's turn}
\end{cases}$$

形式上，最大化玩家的状态 $s$ 的值被递归定义为：

$$V(s) = \begin{cases} 
U(s) & \text{如果 } s \text{ 是终局状态} \\
\max_{a \in A(s)} V(T(s,a)) & \text{如果是玩家的回合} \\
\min_{a \in A(s)} V(T(s,a)) & \text{如果是对手的回合}
\end{cases}$$

where $U(s)$ is the utility function for terminal states (typically +1 for win, 0 for draw, -1 for loss), $A(s)$ is the set of legal actions from state $s$, and $T(s,a)$ is the resulting state after taking action $a$ from state $s$.

其中 $U(s)$ 是终局状态的效用函数（通常胜利为+1，平局为0，失败为-1），$A(s)$ 是从状态 $s$ 开始的合法动作集合，$T(s,a)$ 是从状态 $s$ 采取动作 $a$ 后的结果状态。

The algorithm works by recursively building a game tree. Starting from the current state, it explores all possible move sequences to a predetermined depth or until terminal states are reached. At terminal states, the utility function assigns concrete values. These values propagate upward through the tree, with MAX levels selecting the maximum child value and MIN levels selecting the minimum child value. The optimal move for the current player is the action leading to the child with the highest value.

该算法通过递归构建博弈树来工作。从当前状态开始，它探索所有可能的移动序列直到预定深度或到达终局状态。在终局状态下，效用函数分配具体值。这些值通过树向上传播，MAX层选择最大子节点值，MIN层选择最小子节点值。当前玩家的最优移动是导向具有最高值的子节点的动作。

For games with modest complexity like Tic Tac Toe, Minimax can compute the complete game tree, guaranteeing optimal play. For more complex games like Connect Four, depth-limited search with heuristic evaluation functions becomes necessary, introducing a tradeoff between computational feasibility and decision quality.

对于像井字棋这样适度复杂的游戏，极小极大算法可以计算完整的博弈树，保证最优对弈。对于像四子棋这样更复杂的游戏，深度限制搜索与启发式评估函数变得必要，引入了计算可行性和决策质量之间的权衡。

### 2.2.2 Alpha-Beta Pruning
### 2.2.2 Alpha-Beta剪枝

Alpha-beta pruning is an optimization technique that significantly reduces the number of nodes evaluated by the Minimax algorithm without affecting the final result. It works by maintaining two values, alpha and beta, that represent the minimum score the maximizing player is assured and the maximum score the minimizing player is assured, respectively.

Alpha-beta剪枝是一种优化技术，显著减少了极小极大算法评估的节点数量，而不影响最终结果。它通过维护两个值，alpha和beta，分别代表最大化玩家确保的最小分数和最小化玩家确保的最大分数。

The algorithm prunes branches that cannot influence the final decision. During tree traversal, if a move is found that proves the current position is worse than a previously examined alternative, that branch can be "pruned" without further exploration.

该算法剪掉无法影响最终决策的分支。在树遍历过程中，如果发现一个移动证明当前位置比先前检查的替代方案更差，那么该分支可以被"剪掉"而无需进一步探索。

The pseudocode for the alpha-beta algorithm is:

```
function alphabeta(node, depth, α, β, maximizingPlayer) is
    if depth = 0 or node is a terminal node then
        return the heuristic value of node
    if maximizingPlayer then
        value := −∞
        for each child of node do
            value := max(value, alphabeta(child, depth−1, α, β, FALSE))
            α := max(α, value)
            if α ≥ β then
                break (* β cutoff *)
        return value
    else
        value := +∞
        for each child of node do
            value := min(value, alphabeta(child, depth−1, α, β, TRUE))
            β := min(β, value)
            if β ≤ α then
                break (* α cutoff *)
        return value
```

Alpha-beta剪枝算法的伪代码为：

```
function alphabeta(node, depth, α, β, maximizingPlayer) is
    if depth = 0 or node is a terminal node then
        return the heuristic value of node
    if maximizingPlayer then
        value := −∞
        for each child of node do
            value := max(value, alphabeta(child, depth−1, α, β, FALSE))
            α := max(α, value)
            if α ≥ β then
                break (* β cutoff *)
        return value
    else
        value := +∞
        for each child of node do
            value := min(value, alphabeta(child, depth−1, α, β, TRUE))
            β := min(β, value)
            if β ≤ α then
                break (* α cutoff *)
        return value
```

The efficiency gain from alpha-beta pruning is substantial. In the best case (when moves are ordered optimally), the algorithm examines only O(b^(d/2)) nodes instead of O(b^d) for regular Minimax, where b is the branching factor and d is the search depth. This allows for significantly deeper searches with the same computational resources.

Alpha-beta剪枝带来的效率提升是显著的。在最佳情况下（当移动被最优排序时），算法只检查O(b^(d/2))个节点，而不是常规极小极大算法的O(b^d)，其中b是分支因子，d是搜索深度。这允许在相同的计算资源下进行显著更深的搜索。

### 2.2.3 Application to Tic Tac Toe and Connect Four
### 2.2.3 应用于井字棋和四子棋

In Tic Tac Toe, the complete game tree can be explored using Minimax with alpha-beta pruning. The evaluation function for terminal states is straightforward: +1 for a win, 0 for a draw, and -1 for a loss from the maximizing player's perspective. Non-terminal evaluation is unnecessary as the entire tree can be searched.

在井字棋中，可以使用带有alpha-beta剪枝的极小极大算法探索完整的博弈树。终局状态的评估函数很简单：从最大化玩家的角度看，胜利为+1，平局为0，失败为-1。由于可以搜索整个树，非终局评估是不必要的。

For Connect Four, the much larger state space necessitates depth-limited search with heuristic evaluation functions. Typical heuristic evaluations include:

1. Piece count difference: A simple heuristic comparing the number of pieces each player has placed.
2. Threat analysis: Counting the number of potential winning lines (threats) for each player.
3. Pattern-based evaluation: Assigning values to specific patterns like "three in a row with an open end."
4. Positional value: Assigning higher values to central positions that offer greater connectivity.

对于四子棋，更大的状态空间需要深度限制搜索和启发式评估函数。典型的启发式评估包括：

1. 棋子数量差异：一种简单的启发式方法，比较每个玩家放置的棋子数量。
2. 威胁分析：计算每个玩家潜在获胜线（威胁）的数量。
3. 基于模式的评估：为特定模式（如"一行三个棋子带有一个开放端"）分配值。
4. 位置价值：为提供更大连接性的中心位置分配更高的值。

A common evaluation function for Connect Four combines these elements:

$$eval(s) = w_1 \cdot pieceAdvantage(s) + w_2 \cdot threatCount(s) + w_3 \cdot centerControl(s)$$

where $w_1$, $w_2$, and $w_3$ are weights defining the relative importance of each factor.

四子棋的常见评估函数结合了这些元素：

$$eval(s) = w_1 \cdot pieceAdvantage(s) + w_2 \cdot threatCount(s) + w_3 \cdot centerControl(s)$$

其中 $w_1$、$w_2$ 和 $w_3$ 是定义每个因素相对重要性的权重。

Move ordering significantly impacts alpha-beta pruning efficiency. Examining moves that are likely to be good first increases the chance of early cutoffs. Common ordering heuristics include:
- Examining captures or threats first
- Using iterative deepening to guide move ordering based on previous, shallower searches
- Utilizing the history heuristic to prioritize moves that have caused cutoffs in similar positions

移动排序显著影响alpha-beta剪枝效率。首先检查可能是好的移动增加了早期剪枝的机会。常见的排序启发式包括：
- 首先检查捕获或威胁
- 使用迭代深化基于先前浅层搜索引导移动排序
- 利用历史启发式方法优先考虑在类似位置导致剪枝的移动

In our implementation, alpha-beta pruning with optimized move ordering allows the Minimax algorithm to efficiently find optimal moves in Tic Tac Toe and explore Connect Four to a practical depth (typically 6-8 plies) within reasonable time constraints.

在我们的实现中，带有优化移动排序的alpha-beta剪枝允许极小极大算法在井字棋中有效地找到最优移动，并在合理的时间约束内探索四子棋到实用深度（通常为6-8回合）。

## 2.3 Reinforcement Learning and Q-Learning
## 2.3 强化学习与Q-Learning

### 2.3.1 Markov Decision Processes
### 2.3.1 马尔可夫决策过程

Reinforcement learning addresses the problem of how an agent should take actions in an environment to maximize cumulative reward. The mathematical framework for reinforcement learning is the Markov Decision Process (MDP), defined by the tuple $(S, A, P, R, \gamma)$ where:
- $S$ is the set of states
- $A$ is the set of actions
- $P(s'|s,a)$ is the transition probability from state $s$ to state $s'$ given action $a$
- $R(s,a,s')$ is the immediate reward after transitioning from $s$ to $s'$ via action $a$
- $\gamma \in [0,1]$ is the discount factor weighting future rewards

强化学习解决的问题是代理如何在环境中采取行动以最大化累积奖励。强化学习的数学框架是马尔可夫决策过程（MDP），由元组 $(S, A, P, R, \gamma)$ 定义，其中：
- $S$ 是状态集
- $A$ 是动作集
- $P(s'|s,a)$ 是给定动作 $a$ 时从状态 $s$ 到状态 $s'$ 的转移概率
- $R(s,a,s')$ 是通过动作 $a$ 从 $s$ 转移到 $s'$ 后的即时奖励
- $\gamma \in [0,1]$ 是对未来奖励进行加权的折扣因子

In the context of board games, states correspond to board configurations, actions are legal moves, transitions are deterministic (making $P(s'|s,a)$ either 0 or 1), and rewards typically come at the end of the game (+1 for winning, 0 for drawing, -1 for losing).

在棋类游戏的背景下，状态对应于棋盘配置，动作是合法移动，转移是确定性的（使 $P(s'|s,a)$ 为0或1），奖励通常在游戏结束时给出（胜利为+1，平局为0，失败为-1）。

The goal in reinforcement learning is to find a policy $\pi : S \rightarrow A$ that maximizes the expected cumulative discounted reward:

$$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, \pi(s_t), s_{t+1}) | s_0 = s \right]$$

强化学习的目标是找到一个策略 $\pi : S \rightarrow A$，最大化预期累积折扣奖励：

$$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, \pi(s_t), s_{t+1}) | s_0 = s \right]$$

For deterministic environments like Tic Tac Toe and Connect Four, we can simplify by removing the expectation. The Markov property – that the next state depends only on the current state and action, not on the history – holds perfectly in these games, making them ideal environments for reinforcement learning.

对于像井字棋和四子棋这样的确定性环境，我们可以通过移除期望来简化。马尔可夫性质——下一个状态仅取决于当前状态和动作，而不取决于历史——在这些游戏中完全成立，使它们成为强化学习的理想环境。

### 2.3.2 Q-Learning Algorithm
### 2.3.2 Q-Learning算法

Q-learning is a model-free reinforcement learning algorithm that learns the value of an action in a particular state. These state-action values, denoted as $Q(s,a)$, represent the expected utility of taking action $a$ in state $s$. The optimal policy can then be derived by selecting the action with the highest Q-value in each state.

Q-learning是一种无模型强化学习算法，学习特定状态下动作的价值。这些状态-动作值，表示为 $Q(s,a)$，代表在状态 $s$ 中采取动作 $a$ 的预期效用。然后可以通过在每个状态中选择具有最高Q值的动作来导出最优策略。

The Q-learning update rule is:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right]$$

Q-learning更新规则是：

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right]$$

where:
- $\alpha \in [0,1]$ is the learning rate
- $r_{t+1}$ is the reward received after taking action $a_t$ in state $s_t$
- $\gamma \in [0,1]$ is the discount factor
- $\max_{a} Q(s_{t+1}, a)$ is the maximum Q-value achievable from the next state

其中：
- $\alpha \in [0,1]$ 是学习率
- $r_{t+1}$ 是在状态 $s_t$ 中采取动作 $a_t$ 后收到的奖励
- $\gamma \in [0,1]$ 是折扣因子
- $\max_{a} Q(s_{t+1}, a)$ 是从下一个状态可达到的最大Q值

This update rule adjusts Q-values based on the temporal difference between the predicted value and the observed reward plus the estimated value of the next state. Over time, with sufficient exploration, Q-values converge to their optimal values, denoted $Q^*(s,a)$.

此更新规则基于预测值与观察到的奖励加上下一个状态的估计值之间的时间差异来调整Q值。随着时间的推移，通过足够的探索，Q值收敛到它们的最优值，表示为 $Q^*(s,a)$。

The complete Q-learning algorithm involves:
1. Initialize Q-values for all state-action pairs, typically to zero
2. For each episode:
   a. Initialize state $s$
   b. Until the terminal state is reached:
      i. Choose action $a$ from state $s$ using a policy derived from Q (e.g., ε-greedy)
      ii. Take action $a$, observe reward $r$ and next state $s'$
      iii. Update $Q(s,a)$ using the update rule
      iv. Set $s \leftarrow s'$

完整的Q-learning算法包括：
1. 初始化所有状态-动作对的Q值，通常为零
2. 对于每个回合：
   a. 初始化状态 $s$
   b. 直到达到终局状态：
      i. 使用从Q导出的策略（例如，ε-贪婪）从状态 $s$ 中选择动作 $a$
      ii. 执行动作 $a$，观察奖励 $r$ 和下一个状态 $s'$
      iii. 使用更新规则更新 $Q(s,a)$
      iv. 设置 $s \leftarrow s'$

When applying Q-learning to board games like Tic Tac Toe and Connect Four, several considerations become important:

1. State representation: The raw board state can be encoded as a tuple or string, but for Connect Four, the state space becomes too large for tabular representation, requiring function approximation or state abstraction techniques.

2. Reward structure: For board games, the reward structure is typically sparse – rewards are only given at the end of the game. This can be addressed by introducing intermediate rewards based on board evaluation or by using eligibility traces to propagate terminal rewards backward.

3. Training through self-play: The agent can play against itself, learning from both sides of the game. This is particularly effective for zero-sum games where optimal play against an optimal opponent leads to a minimax strategy.

在将Q-learning应用于井字棋和四子棋等棋类游戏时，几个考虑因素变得重要：

1. 状态表示：原始棋盘状态可以编码为元组或字符串，但对于四子棋，状态空间变得太大，无法进行表格表示，需要函数近似或状态抽象技术。

2. 奖励结构：对于棋类游戏，奖励结构通常是稀疏的——只在游戏结束时给予奖励。这可以通过基于棋盘评估引入中间奖励或使用资格迹（eligibility traces）向后传播终局奖励来解决。

3. 通过自我对弈进行训练：代理可以与自己对弈，从游戏的双方学习。这对于零和博弈特别有效，在这种情况下，对抗最优对手的最优对弈导致极小极大策略。

### 2.3.3 Exploration vs. Exploitation Balance
### 2.3.3 探索与利用平衡

A fundamental challenge in reinforcement learning is balancing exploration (trying new actions to discover their values) with exploitation (selecting the best-known action based on current estimates). This is crucial for Q-learning to converge to optimal policies.

强化学习中的一个基本挑战是平衡探索（尝试新动作以发现其价值）与利用（基于当前估计选择已知最佳动作）。这对于Q-learning收敛到最优策略至关重要。

The ε-greedy strategy is a common approach:
- With probability ε, choose a random action (exploration)
- With probability 1-ε, choose the action with the highest Q-value (exploitation)

ε-贪婪策略是一种常见的方法：
- 以概率ε选择随机动作（探索）
- 以概率1-ε选择具有最高Q值的动作（利用）

The value of ε is typically high initially and decreases over time, allowing the agent to explore extensively early on and gradually shift towards exploitation as knowledge improves. This can be implemented using an annealing schedule:

$$\varepsilon = \varepsilon_{min} + (\varepsilon_{max} - \varepsilon_{min}) \cdot e^{-decay \cdot episode}$$

ε的值通常在初始时很高，并随着时间的推移而降低，允许代理在早期广泛探索，并随着知识的改善逐渐转向利用。这可以使用退火计划实现：

$$\varepsilon = \varepsilon_{min} + (\varepsilon_{max} - \varepsilon_{min}) \cdot e^{-decay \cdot episode}$$

More sophisticated exploration strategies include:

1. Boltzmann exploration (softmax): Actions are selected with probabilities proportional to their expected values:
   $$P(a|s) = \frac{e^{Q(s,a)/\tau}}{\sum_{a'} e^{Q(s,a')/\tau}}$$
   where $\tau$ is a temperature parameter controlling exploration.

2. Upper Confidence Bound (UCB): Incorporates uncertainty in Q-value estimates:
   $$UCB(s,a) = Q(s,a) + c \sqrt{\frac{\ln N(s)}{N(s,a)}}$$
   where $N(s)$ is the number of visits to state $s$, $N(s,a)$ is the number of times action $a$ was taken in state $s$, and $c$ is an exploration parameter.

3. Optimistic initialization: Initialize Q-values to high values, encouraging exploration of unvisited state-action pairs.

更复杂的探索策略包括：

1. 玻尔兹曼探索（softmax）：选择动作的概率与其预期值成比例：
   $$P(a|s) = \frac{e^{Q(s,a)/\tau}}{\sum_{a'} e^{Q(s,a')/\tau}}$$
   其中 $\tau$ 是控制探索的温度参数。

2. 上置信界（UCB）：包含Q值估计中的不确定性：
   $$UCB(s,a) = Q(s,a) + c \sqrt{\frac{\ln N(s)}{N(s,a)}}$$
   其中 $N(s)$ 是访问状态 $s$ 的次数，$N(s,a)$ 是在状态 $s$ 中采取动作 $a$ 的次数，$c$ 是探索参数。

3. 乐观初始化：将Q值初始化为高值，鼓励探索未访问的状态-动作对。

In our implementation, we use an ε-greedy strategy with annealing for Tic Tac Toe. For Connect Four, due to its larger state space, we combine ε-greedy exploration with domain-specific heuristics to guide exploration toward promising regions of the state space.

在我们的实现中，我们对井字棋使用带有退火的ε-贪婪策略。对于四子棋，由于其更大的状态空间，我们将ε-贪婪探索与特定领域的启发式方法相结合，引导探索朝向状态空间的有希望区域。

## 2.4 Deep Learning in Board Games
## 2.4 棋类游戏中的深度学习

### 2.4.1 Convolutional Neural Networks
### 2.4.1 卷积神经网络

Convolutional Neural Networks (CNNs) have revolutionized board game AI by providing powerful function approximators that can learn to evaluate complex game states. CNNs are particularly well-suited for board games due to their ability to capture spatial patterns and hierarchical features.

卷积神经网络（CNNs）通过提供强大的函数近似器，可以学习评估复杂的游戏状态，从而彻底改变了棋类游戏AI。由于能够捕获空间模式和层次特征，CNNs特别适合棋类游戏。

The basic structure of a CNN for board games includes:

1. Input layer: The board state is typically represented as a stack of feature planes. For Connect Four, this might include separate planes for the player's pieces, opponent's pieces, and potentially additional features like "last move" or "turn indicator."

2. Convolutional layers: These apply learned filters across the board, detecting patterns such as potential winning lines or threats. Each convolutional layer is defined as:
   $$Z^l_{i,j,k} = \sum_{m,n,c} W^l_{m,n,c,k} \cdot X^{l-1}_{i+m, j+n, c} + b^l_k$$
   where $Z^l_{i,j,k}$ is the output at position $(i,j)$ for filter $k$ in layer $l$, $W^l$ are the weights, $X^{l-1}$ is the input from the previous layer, and $b^l_k$ is the bias term.

3. Activation function: Typically ReLU (Rectified Linear Unit):
   $$X^l_{i,j,k} = \max(0, Z^l_{i,j,k})$$

4. Pooling layers: Optional downsampling operations that reduce dimensionality while preserving important features.

5. Fully connected layers: Process the high-level features extracted by convolutional layers.

6. Output layer: For board games, typically produces:
   - Policy head: A probability distribution over possible moves
   - Value head: An evaluation of the current position (typically in the range [-1, 1])

棋类游戏CNN的基本结构包括：

1. 输入层：棋盘状态通常表示为特征平面的堆叠。对于四子棋，这可能包括玩家棋子、对手棋子的单独平面，以及可能的额外特征，如"最后一步"或"回合指示器"。

2. 卷积层：这些在棋盘上应用学习的过滤器，检测诸如潜在获胜线或威胁等模式。每个卷积层定义为：
   $$Z^l_{i,j,k} = \sum_{m,n,c} W^l_{m,n,c,k} \cdot X^{l-1}_{i+m, j+n, c} + b^l_k$$
   其中 $Z^l_{i,j,k}$ 是层 $l$ 中过滤器 $k$ 在位置 $(i,j)$ 的输出，$W^l$ 是权重，$X^{l-1}$ 是来自上一层的输入，$b^l_k$ 是偏置项。

3. 激活函数：通常是ReLU（修正线性单元）：
   $$X^l_{i,j,k} = \max(0, Z^l_{i,j,k})$$

4. 池化层：可选的下采样操作，在保留重要特征的同时减少维度。

5. 全连接层：处理卷积层提取的高级特征。

6. 输出层：对于棋类游戏，通常产生：
   - 策略头：可能移动的概率分布
   - 价值头：当前位置的评估（通常在范围[-1, 1]内）

For Connect Four, our implementation uses a simplified architecture with:
- 3 input planes (own pieces, opponent pieces, turn indicator)
- 4 convolutional layers with 3×3 filters
- 2 fully connected layers
- Dual output heads for policy and value

对于四子棋，我们的实现使用简化的架构：
- 3个输入平面（自己的棋子，对手的棋子，回合指示器）
- 4个带有3×3过滤器的卷积层
- 2个全连接层
- 策略和价值的双输出头

The network is trained using a combination of supervised learning (from expert games when available) and reinforcement learning through self-play.

网络使用监督学习（来自可用的专家游戏）和通过自我对弈的强化学习的组合进行训练。

### 2.4.2 Monte Carlo Tree Search
### 2.4.2 蒙特卡洛树搜索

Monte Carlo Tree Search (MCTS) is a heuristic search algorithm that combines tree search with random sampling. Unlike Minimax, which explores the game tree deterministically, MCTS uses statistical sampling to evaluate nodes and guide search toward promising areas.

蒙特卡洛树搜索（MCTS）是一种启发式搜索算法，结合了树搜索和随机抽样。与确定性地探索博弈树的极小极大算法不同，MCTS使用统计抽样来评估节点并引导搜索朝向有希望的区域。

The standard MCTS algorithm consists of four phases, repeated for multiple iterations:

1. Selection: Starting from the root, the tree is traversed by selecting the most promising child nodes according to a tree policy, typically using the Upper Confidence Bound for Trees (UCT) formula:
   $$UCT(s,a) = \frac{Q(s,a)}{N(s,a)} + c \cdot \sqrt{\frac{\ln N(s)}{N(s,a)}}$$
   where $Q(s,a)$ is the total reward from action $a$ in state $s$, $N(s,a)$ is the visit count, $N(s)$ is the parent node visit count, and $c$ is an exploration parameter.

2. Expansion: When a leaf node is reached, one or more child nodes are added to expand the tree.

3. Simulation (Rollout): From the newly added node, a simulation is played out to the end of the game using a default policy (often random play or simple heuristics).

4. Backpropagation: The result of the simulation is propagated back up the tree, updating statistics (visit counts and win rates) for each node traversed.

标准的MCTS算法由四个阶段组成，重复多次迭代：

1. 选择：从根节点开始，通过根据树策略选择最有希望的子节点来遍历树，通常使用树的上置信界（UCT）公式：
   $$UCT(s,a) = \frac{Q(s,a)}{N(s,a)} + c \cdot \sqrt{\frac{\ln N(s)}{N(s,a)}}$$
   其中 $Q(s,a)$ 是状态 $s$ 中动作 $a$ 的总奖励，$N(s,a)$ 是访问计数，$N(s)$ 是父节点访问计数，$c$ 是探索参数。

2. 扩展：当到达叶节点时，添加一个或多个子节点以扩展树。

3. 模拟（展开）：从新添加的节点开始，使用默认策略（通常是随机游戏或简单启发式）进行模拟直到游戏结束。

4. 反向传播：模拟的结果向上传播回树，更新每个遍历节点的统计数据（访问计数和胜率）。

After a predetermined number of iterations, the algorithm selects the action with the highest visit count or highest average value.

在预定数量的迭代后，算法选择具有最高访问计数或最高平均值的动作。

MCTS offers several advantages over traditional search algorithms:
- It does not require a heuristic evaluation function
- It can be interrupted at any time to return the best move found so far
- It handles high branching factors well through selective tree growth
- It naturally balances exploration and exploitation

MCTS比传统搜索算法提供几个优势：
- 它不需要启发式评估函数
- 它可以在任何时候中断以返回迄今为止发现的最佳移动
- 它通过选择性树生长很好地处理高分支因子
- 它自然地平衡探索和利用

For Connect Four, pure MCTS can achieve strong play with sufficient simulation budget, but its performance is significantly enhanced when combined with neural networks.

对于四子棋，纯MCTS可以通过足够的模拟预算实现强大的对弈，但当与神经网络结合时，其性能显著提高。

### 2.4.3 Integration of CNN and MCTS
### 2.4.3 CNN与MCTS的结合

The integration of CNNs with MCTS represents a breakthrough in game AI, exemplified by AlphaGo and AlphaZero. This hybrid approach leverages the strengths of both methods: CNNs provide strategic understanding and pattern recognition, while MCTS adds tactical calculation and exploration.

CNN与MCTS的结合代表了博弈AI的突破，以AlphaGo和AlphaZero为例。这种混合方法利用了两种方法的优势：CNN提供战略理解和模式识别，而MCTS添加了战术计算和探索。

The key modifications to standard MCTS include:

1. Selection: The UCT formula is enhanced with policy network predictions:
   $$UCT(s,a) = \frac{Q(s,a)}{N(s,a)} + c \cdot P(s,a) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)}$$
   where $P(s,a)$ is the policy network's probability for action $a$ in state $s$.

2. Evaluation: Instead of random rollouts, the value network directly evaluates leaf nodes:
   $$V(s) = \text{ValueNetwork}(s)$$

3. Training: The neural networks are trained from self-play data generated by MCTS:
   - Policy targets: The empirical distribution of MCTS visit counts
   - Value targets: The actual game outcomes

对标准MCTS的关键修改包括：

1. 选择：UCT公式通过策略网络预测增强：
   $$UCT(s,a) = \frac{Q(s,a)}{N(s,a)} + c \cdot P(s,a) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)}$$
   其中 $P(s,a)$ 是策略网络对状态 $s$ 中动作 $a$ 的概率。

2. 评估：不是随机展开，价值网络直接评估叶节点：
   $$V(s) = \text{ValueNetwork}(s)$$

3. 训练：神经网络从MCTS生成的自我对弈数据中训练：
   - 策略目标：MCTS访问计数的经验分布
   - 价值目标：实际游戏结果

The training process follows an iterative improvement loop:
1. Neural networks guide MCTS during self-play
2. Self-play games generate training data
3. Neural networks are updated based on this data
4. Improved networks are used for the next iteration

训练过程遵循迭代改进循环：
1. 神经网络在自我对弈期间指导MCTS
2. 自我对弈游戏生成训练数据
3. 神经网络根据这些数据更新
4. 改进的网络用于下一次迭代

For Connect Four, our implementation uses a simplified version of this approach, with several adaptations for computational efficiency:
- Smaller neural network architecture
- Reduced MCTS simulation count (1000 simulations per move vs. tens of thousands in AlphaZero)
- Enhanced exploration to compensate for lower simulation counts
- Progressive widening to manage the branching factor

对于四子棋，我们的实现使用了这种方法的简化版本，为了计算效率做了几个调整：
- 较小的神经网络架构
- 减少MCTS模拟计数（每步1000次模拟，而AlphaZero中是数万次）
- 增强探索以补偿较低的模拟计数
- 渐进式扩展以管理分支因子

This approach significantly outperforms both pure MCTS and traditional Minimax with alpha-beta pruning for Connect Four, demonstrating the power of integrating deep learning with tree search methods.

这种方法显著优于四子棋的纯MCTS和传统的带alpha-beta剪枝的极小极大算法，展示了将深度学习与树搜索方法集成的威力。
