# Graph2Plan: Learning Floorplan Generation from Layout Graphs RUIZHEN HU, Shenzhen University ZEYU HUANG, Shenzhen University YUHAN TANG, Shenzhen University OLIVER VAN KAICK, Carle

- PDF: [hu2020graph2plan.pdf](../../hu2020graph2plan.pdf)
- Topic: graph_conditioning_layout

## Abstract / Core Idea
Graph2Plan: Learning Floorplan Generation from Layout Graphs RUIZHEN HU, Shenzhen University ZEYU HUANG, Shenzhen University YUHAN TANG, Shenzhen University OLIVER VAN KAICK, Carleton University HAO ZHANG, Simon Fraser University HUI HUANG∗, Shenzhen University Bedroom Bathroom Balcony 1+ 2 1 ... ... ... (a) Input building boundary. (b) Generated floorplans. (c) After adding room counts. (d) After adding room connectivity. (e) After layout graph editing. Kitchen Kitchen Living Room Living Room Living Room Living Room Living Room Living Room Living Room Bedroom BedroomBedroom Bedroom Bedroom Bedroom Br Br Br Br Bedroom Bedroom Bathroom Bathroom Bath Bath Bath Bath Balcony Balcony Bal Bal Kit Kit Balcony Balcony Bal. Bal. Kitchen Kitchen Kitchen Bedroom Bedroom Bedroom Bathroom Balcony Balcony Fig. 1. Our deep neural network Graph2Plan is a learning framework for automated floorplan generation from layout graphs. The trained network can generate floorplans based on an input building boundary only (a-b), like in previous works. In addition, we allow users to add a variety of constraints such as room counts (c), room connectivity (d), and other layout graph edits. Multiple generate

## Method Signals
- Keywords phat hien: graph, layout, constraint

## Conclusion / Findings
AND FUTURE WORK We introduce the first deep learning framework for floorplan gener- ation that enables user-in-the-loop modeling. The users can specify their design goals with constraints that guide the retrieval of layout graphs from a large dataset of floorplans, and can further refine the constraints by editing the layout graphs. The layout graphs specify the desired numbers and types of rooms along with the room adjacencies, and directly guide the floorplan generation. We demonstrated with a series of experiments that this framework al- lows users to generate a variety of floorplans from the same input boundary, and fine-tune the results by editing the layout graphs. In addition, a quantitative evaluation shows that the floorplans are similar to training examples and thus also tend to follow the design principles learned from the dataset of floorplans. Limitations. As this work is a first step in the direction of user- guide floorplan generation, it has certain limitations. Although the layout graphs model the user preferences in terms of desired room (a) Input boundary (b) Retrieved floorplan (c) Retrieved graph (d) Adjusted graph (e) Final floorplan Fig. 19. Failure case. Given the input boundary (a), if the retrieved floorplan (b) has quite a different boundary, then the corresponding retrieved graph (c) needs to be sufficiently adjusted (d) to fit into the input boundar

## Relevance To KLTN
- Bai nay duoc xep vao nhom graph_conditioning_layout trong pipeline neural-symbolic topology-first.
- Dung de doi chieu voi khoi tuong ung trong docs va Chapter 3/4.
