# Generative Adversarial Network Rooms in Generative Graph Grammar Dungeons for The Legend of Zelda

- PDF: [generative_adversarial_network_rooms_in_generative_graph_grammar_dungeons_for_th.pdf](../../generative_adversarial_network_rooms_in_generative_graph_grammar_dungeons_for_th.pdf)
- Topic: pcg_mission_zelda
- Reference IDs: arxiv:2001.05065

## Abstract / Core Idea
Generative Adversarial Networks (GANs) have demonstrated their ability to learn patterns in data and produce new exemplars similar to, but different from, their training set in several domains, including video games. However, GANs have a fixed output size, so creating levels of arbitrary size for a dungeon crawling game is difficult. GANs also have trouble encoding semantic requirements that make levels interesting and playable. This paper combines a GAN approach to generating individual rooms with a graph grammar approach to combining rooms into a dungeon. The GAN captures design principles of individual rooms, but the graph grammar organizes rooms into a global layout with a sequence of obstacles determined by a designer. Room data from The Legend of Zelda is used to train the GAN. This approach is validated by a user study, showing that GAN dungeons are as enjoyable to play as a level from the original game, and levels generated with a graph grammar alone. However, GAN dungeons have rooms considered more complex, and plain graph grammar's dungeons are considered least complex and challenging. Only the GAN approach creates an extensive supply of both layouts and rooms, where rooms span across the spectrum of those seen in the training set to new creations merging design principles from multiple rooms.

## Method Signals
- Keywords phat hien: graph, layout, dungeon, zelda

## Conclusion / Findings
dungeons have stairs to standalone rooms that are not part of the main map layout. Stairs are excluded from dungeons in this study. Many interesting items can be collected in the game, but only a few are relevant to this paper: keys, hearts, bombs, and the raft. Hearts replenish a player’s health. Bombs allow the player to blow up walls to reveal hidden doors or kill enemies. The raft item allows players to move across one water tile. It is introduced in Dungeon 4-1 (4th dungeon of Quest 1, Fig. 1) and used throughout the rest of the game. Data about Zelda levels was obtained from the Video Game Level Corpus (VGLC [8]). This data provides text representations of the tiles present in each dungeon. Details of this representation, and how it maps to the one used in this paper, are in Table I. There are many symbols from the VGLC data, but since many of these tiles serve the same purpose as others, the tile training set is simpliﬁed. IV. D UNGEON GENERATION A GAN is trained to generate individual rooms, which can then be combined into d S E s e e e SL K L e l k k l e E T e t e e e t e e t SL L l K k SL SL e sl P p Fig. 8: Graph Grammar Rules. This set of rules deﬁnes how symbols (Orange/Uppercase) can map to terminals (Blue/Lowercase) in a ﬁnal dungeon. Though the set is small, and some symbol pairs only have one possible transformation, there is enough variety in the rule set to create many different dungeons, especially when combined with different room placements and layouts.

## Relevance To KLTN
- Bai nay duoc xep vao nhom pcg_mission_zelda trong pipeline neural-symbolic topology-first.
- Dung de doi chieu voi khoi tuong ung trong docs va Chapter 3/4.
