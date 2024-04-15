# Related Work.

This study investigates approaches to promote competitiveness in
Dynamic-MTCs, where agents work together with teammates and compete with
agents from other different teams.

Similar studies can be traced back to early work that explores self-play
using classic evolutionary algorithms
[@paredis1995; @pollack1997; @rosin1995; @stanley2004]. In the field of
MARL, many prior works focus on 1-agent-vs-1-agent or 1-agent-vs-many
tasks, such as Backgammon [@tesauro1995], Go [@silver2016], IPD
[@pmlr-v162-lu22d], global Starcraft control [@vinyals2019],
Kick-and-Defend [@bansal2017] and Sumo [@bansal2017]. Many inspiring
techniques are used in these studies, such as latent representation
learning [@xie2021learning], opponent modeling [@pmlr-v162-lu22d] and
curriculum exploration guiding [@bansal2017]. These works reveal that
sophisticated competition policies can be developed in self-play between
agents.

On the other hand, pure cooperative MARL has gained growing recognition
in recent years. Such as SMAC v1 [@samvelyan2019starcraft], a multiagent
version of the Starcraft game. To achieve the goal of achieving closer
cooperation within a multiagent team, prior works have explored diverse
approaches including but not limited to reward decomposition
[@sunehag2017value; @rashid2018qmix], mean field
implementation[@yang2018mean], network structure enhancement
[@wu2021multi] and bidirectional action-dependency [@li2022ace] etc.
MAPPO [@yu2021surprising] is a representative method among these studies
due to its simplicity and effectiveness, and it is still one of the SOTA
methods in many benchmarks. Many of these works use stationary and
preprogrammed AIs to play the role of opponents and can also simulate
competitive environments.

The topic of this paper, MARL in dynamic team-vs-team competition,
combines the cooperative feature with competitiveness self-play. To this
extent, our work inherits from prior works such as Capture-the-Flag
[@Jaderberg859], Hide-and-Seek [@baker2019emergent] and 2-vs-2 football
[@liu2019emergent]. However, while these works emphasize the study on
the emergence of cooperative behaviors [@liu2019emergent] and the
astonishing long-term policy shift during 2-team competition
[@baker2019emergent]. We first formulate our work by extending the
2-team competition to arbitrary N-team competition. Next, we ask a
natural question about how to obtain a leading position over opponents
aside from benefiting from joint policy emergence. Only a few studies
such as [@ma2021opponent] discuss similar problems under this specific
dynamic team-vs-team background, probably because it is very difficult
at the code-implementation level to simultaneously manage, optimize and
keep track of multiple different teams powered with different and
independently running RL optimizers.

Many different approaches are developed in prior studies that aim to
promote the competitiveness of the agent(s), especially in the studies
on 1-vs-1 games. Opponent portraiting
[@ma2021opponent; @xie2021learning] methods allow the agent(s) to learn
more about the opponent(s), using latent representations to predict the
behaviors of the opponent(s). However, modeling a whole team of
opponents (team-vs-team) is much more sophisticated than modeling just
one agent (1-vs-1), and this portraiting strategy can be countered by
deliberately displaying fraudulent actions. Meta reinforcement learning
[@pmlr-v162-lu22d; @xu2018meta] offers promising results in small scale
environment when the number of agents is relatively smaller, but the
policy update cycle is significantly extended because each meta-step is
usually an entire episode of the underlying inner game. When each side
has an equal amount of inner game samples, meta learners can suffer a
long period of disadvantage before leading the game. Population-based
methods and classic genetic algorithms are also strong solutions for
obtaining competition advantages, however, they have similar sample
efficiency problems and potentially require a large cluster of servers
[@Jaderberg859] to simulate the population. In comparison, AutoRL
[@parker2022automated] is a better choice considering the balance of
efficiency and effectiveness. We use CASMOPOLITAN (BayesOpt for
Categorical and Mixed Search Spaces) [@wan2021think], a Bayesian
Optimization [@snoek2012practical] algorithm, to carrier out AutoRL to
search for a set of feedback fuzzy logic to promote team
competitiveness.

# Supplementary Material. {#supplementary-material. .unnumbered}
