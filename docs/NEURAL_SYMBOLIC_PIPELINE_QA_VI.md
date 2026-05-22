# Q&A : 7-Block Neural-Symbolic Pipeline

## 1) Dữ liệu và số liệu thực tế

### Q1. Dữ liệu này được lấy từ nguồn nào?

Trong repo hiện tại, bộ dữ liệu gốc là Zelda dungeon corpus thuộc VGLC. Dữ liệu này đã được VGLC chuẩn hóa và đi kèm cả graph topo lẫn bảng nhãn tương ứng. Vì vậy, đây không phải là dữ liệu tự sinh trong dự án mà là tập Zelda được chuẩn bị từ nguồn VGLC.

Về mặt tổ chức, dữ liệu được sử dụng trên ba lớp: map lớn ở dạng text, graph ở dạng DOT, và bảng ánh xạ nhãn để diễn giải node/edge theo nghĩa ngữ nghĩa. Khi đi vào pipeline, các thành phần này được chuyển thành room-level samples và conditioning metadata tương ứng.

Nguồn gốc tham khảo:

- GitHub: https://github.com/TheVGLC/TheVGLC
- Paper: https://arxiv.org/abs/1606.07487

### Q2. Có tổng cộng bao nhiêu bản đồ lớn (Map level)?

Có 18 map lớn. Dữ liệu gồm 9 dungeon chính, mỗi dungeon có 2 biến thể, nên tổng cộng là 9 × 2 = 18.

### Q3. Nếu cắt các bản đồ lớn ra thành từng phòng nhỏ, thì tổng số lượng phòng là bao nhiêu?

Tổng số room-level samples là 459. Đây là số lượng phòng thực tế mà loader tạo ra sau khi đi qua toàn bộ 9 dungeon × 2 variant.

| Dungeon         |     Variant 1 |     Variant 2 |         Tổng |
| --------------- | ------------: | ------------: | ------------: |
| 1               |            17 |            14 |            31 |
| 2               |            18 |            20 |            38 |
| 3               |            18 |            11 |            29 |
| 4               |            20 |            30 |            50 |
| 5               |            23 |            18 |            41 |
| 6               |            25 |            22 |            47 |
| 7               |            33 |            27 |            60 |
| 8               |            25 |            35 |            60 |
| 9               |            57 |            46 |           103 |
| **Tổng** | **236** | **223** | **459** |

### Q4. Tỷ lệ chia tập dữ liệu: có bao nhiêu Room map dùng để Train, bao nhiêu dùng để Test?

Hiện tại, repo chưa định nghĩa một test split cố định dùng chung cho toàn bộ pipeline. Ở stage VQ-VAE, dữ liệu 459 phòng được chia train/validation bằng `validation_fraction = 0.1`, tương ứng 413 mẫu train và 46 mẫu validation theo cấu hình mặc định.

Cần lưu ý rằng diffusion stage chưa triển khai một test split độc lập theo cùng cơ chế này. Do đó, nếu mô tả đúng theo code hiện hành, nên xem đây là thiết kế có train/validation theo từng stage, nhưng chưa có test split riêng và thống nhất cho toàn hệ thống.

### Q5. Dữ liệu room-level này được đọc như thế?

Bộ nạp dữ liệu duyệt qua toàn bộ 9 dungeon × 2 variant, trích xuất từng room đã được tách riêng, rồi lưu semantic grid của room đó thành một sample huấn luyện. Khi bật `load_graphs=True`, mỗi room còn được ghép với một dict conditioning chứa node features, edge index, edge features, TPE, topology map, boundary constraints, room position, neighbor maps và các cờ ngữ nghĩa khác.

Như vậy, dataset không được huấn luyện trực tiếp trên 18 map lớn. Thay vào đó, mô hình học trên 459 phòng đã tách, trong đó mỗi phòng có thể đi kèm một gói graph-conditioning riêng.

## 2) Ví dụ cụ thể: Dungeon 1 Variant 1

Dưới đây là một ví dụ thực nghiệm từ dungeon 1 variant 1 nhằm minh họa rõ từng biến đầu ra.

```python
adapter = ZeldaDungeonAdapter(data_root="Data/The Legend of Zelda")
dungeon = adapter.load_dungeon(1, 1)
first_pos = next(iter(dungeon.rooms.keys()))
room = dungeon.rooms[first_pos]

dataset = ZeldaRoomDataset(data_dir="Data/The Legend of Zelda", normalize=True, load_graphs=True)
grid, graph = dataset[0]
```

| Biến                                      | Giá trị ví dụ                               | Ý nghĩa                                     |
| ------------------------------------------ | ----------------------------------------------- | --------------------------------------------- |
| `dungeon`                                | `Dungeon`, 17 rooms                           | Toàn bộ dungeon sau khi ghép xong          |
| `first_pos`                              | `(0, 2)`                                      | Tọa độ room đầu tiên trong dungeon này |
| `room`                                   | room tại `(0, 2)`                            | Đối tượng room đang được đọc        |
| `room.graph_node_id`                     | `17`                                          | Node graph đã được ghép với room       |
| `room.node_label`                        | `e,k`                                         | Nhãn node trong mission graph                |
| `grid`                                   | tensor `torch.Size([1, 16, 11])`, `float32` | Semantic grid của room                       |
| `graph`                                  | `dict`                                        | Gói conditioning đi kèm room               |
| `graph['node_features']`                 | `torch.Size([18, 14])`                        | Đặc trưng của 18 node                     |
| `graph['edge_index']`                    | `torch.Size([2, 38])`                         | Danh sách cạnh graph                        |
| `graph['edge_attr']`                     | `torch.Size([38])`                            | Loại cạnh cho từng edge                    |
| `graph['edge_features']`                 | `torch.Size([38, 16])`                        | Đặc trưng cạnh                            |
| `graph['tpe']`                           | `torch.Size([18, 8])`                         | Topological positional encoding               |
| `graph['node_positions']`                | `torch.Size([18, 2])`                         | Vị trí node theo room coordinate            |
| `graph['room_topology_map']`             | `torch.Size([54, 16, 11])`                    | Topology supervision map                      |
| `graph['boundary_constraints']`          | `torch.Size([8])`                             | Ràng buộc biên của room                   |
| `graph['room_position']`                 | `torch.Size([2])`                             | Vị trí của room trong dungeon              |
| `graph['neighbor_maps']`                 | `dict(4)`                                     | Bản đồ phòng lân cận theo 4 hướng     |
| `graph['num_nodes']`                     | `18`                                          | Tổng số node trong graph                    |
| `graph['num_edges']`                     | `38`                                          | Tổng số cạnh trong graph                   |
| `graph['start_node_id']`                 | `7`                                           | Node khởi đầu trong graph                  |
| `graph['current_node_idx']`              | `16`                                          | Chỉ số node hiện tại trong mapping        |
| `graph['node_to_idx']`                   | `dict(18)`                                    | Bảng ánh xạ node id sang chỉ số tensor   |
| `graph['has_puzzle']`                    | `False`                                       | Cờ phòng có puzzle hay không              |
| `graph['puzzle_room_structure_enabled']` | `False`                                       | Cờ topology puzzle-room                      |
| `graph['puzzle_stage_condition']`        | `dict(6)`                                     | Metadata cho stage-based puzzle planning      |

Điểm quan trọng ở ví dụ này là `dataset[0]` trả về một tuple gồm hai phần, không phải một dict đơn lẻ. Phần đầu là tensor của room, phần sau là dict graph-conditioning đã được align với room đó.

## 3) Parse graph -> align room tiles diễn ra thế nào?

Đây là giai đoạn dễ gây nhầm lẫn nhất, vì code không chỉ đọc graph mà còn khớp node graph với vị trí room thực tế, sau đó dựng các tensor supervision để mô hình học được quan hệ giữa topology và tile.

```python
room_to_node, node_to_room = match_rooms_to_graph_impl(...)
rooms[room_pos].graph_node_id = node_id
rooms[room_pos].node_label = node_data.get("label", "")

start = nearest_walkable_point(room_grid, heuristic_anchor)
room_topology_map = build_room_topology_condition_map(...)
return tensor, graph_dict
```

Luồng xử lý này có thể được diễn giải theo bốn bước. Bước thứ nhất là parse graph: node và edge của DOT graph được chuyển thành `node_features`, `edge_index`, `edge_features` và TPE. Bước thứ hai là room matching: graph node được gán vào `room position` bằng BFS kết hợp cost matching, rồi room nhận thêm `graph_node_id` và `node_label`. Bước thứ ba là align tile: start, goal, door và các anchor khác được snap về tile đi bộ gần nhất bằng nearest-walkable search. Bước cuối là build conditioning: từ các anchor và quan hệ graph-room, code sinh ra `room_topology_map`, `boundary_constraints`, `neighbor_maps` và các metadata khác.

Như vậy, đây không phải là bước sửa map gốc theo nghĩa tái vẽ tile, mà là bước tạo supervision đã được căn chỉnh giữa graph trừu tượng và semantic grid 16 × 11.

## 4) Diễn giải sơ đồ theo cụm a/b/c/d

### Q6. Nếu chia hình thành 4 cụm a/b/c/d thì hiểu thế nào?

Có thể diễn giải sơ đồ theo 4 lớp chức năng lớn. Cụm (a) là phần nhập dữ liệu và dựng topology ban đầu, tức Block 0 và Block I. Cụm (b) là lõi học máy: Block II, III, IV và V, nơi room được nén thành latent, được điều kiện hóa bằng graph, rồi được diffusion sinh lại dưới sự dẫn hướng của logic. Cụm (c) là Block VI, nơi kết quả neural được sửa bằng symbolic repair nếu còn lỗi solvability. Cụm (d) là Block VII, nơi toàn bộ dungeon được kiểm tra, chấm điểm và ghi nhận chất lượng.

Nhìn ở mức hệ thống, sơ đồ này không biểu diễn một mô hình đơn lẻ mà mô tả một chuỗi xử lý nhiều tầng: dữ liệu thô được parse và căn chỉnh, sau đó được chuyển thành biểu diễn rời rạc, điều kiện hóa, sinh latent, hiệu chỉnh bằng logic và cuối cùng được đánh giá.

| Cụm | Thành phần chính  | Vai trò                                                                              |
| ---- | -------------------- | ------------------------------------------------------------------------------------- |
| (a)  | Block 0, Block I     | Đưa dữ liệu Zelda thô vào đúng cấu trúc graph/room để model có thể học |
| (b)  | Block II, III, IV, V | Học biểu diễn room, nén latent, điều kiện hóa và sinh mẫu có kiểm soát   |
| (c)  | Block VI             | Sửa room hoặc dungeon bị lỗi bằng quy tắc symbolic và WFC                      |
| (d)  | Block VII            | Đánh giá độ chơi được, độ hợp lệ và chất lượng tổng thể            |

### Q7. Toàn bộ mô hình gồm những thành phần nào?

| Block                                     | Input chính                                          | Xử lý chính                                                 | Output                           | Khi dùng                                       |
| ----------------------------------------- | ----------------------------------------------------- | -------------------------------------------------------------- | -------------------------------- | ----------------------------------------------- |
| 0 - Data Adapter                          | file dungeon, graph, label map                        | parse, match room, align tile, tạo metadata                   | room sample + graph conditioning | khi nạp dữ liệu và tạo sample huấn luyện |
| I - Search-based Mission Graph Generation | mục tiêu topology, số room, quan hệ key/lock/boss | sinh hoặc điều phối mission graph bằng search/ràng buộc | mission graph                    | khi cần tạo cấu trúc dungeon ở mức graph  |
| II - Semantic VQ-VAE Tokenizer            | semantic room tensor                                  | encoder, vector quantization, decoder                          | latent room tokens               | khi học biểu diễn room rời rạc             |
| III - Dual-Stream Conditioning            | neighbor rooms + graph                                | local stream, global stream, fusion                            | conditioning vector              | khi cần ngữ cảnh cho diffusion               |
| IV - Latent Diffusion                     | noisy latent + conditioning                           | U-Net denoise, attention, topology injection                   | predicted latent/noise           | khi sinh room mới                              |
| V - LogicNet Steering                     | latent dự đoán + graph data                        | tính logic loss / gradient guidance                           | guidance gradient                | khi muốn ép mẫu hợp luật hơn              |
| VI - Symbolic Repair & Overlay            | room lỗi hoặc dungeon lỗi                          | path analysis, entropy reset, WFC, propagation                 | room/dungeon đã sửa           | khi validator báo lỗi                         |
| VII - Validation & QD Evaluation          | dungeon sau sinh                                      | kiểm tra solvability, quality-diversity, report               | metrics / verdict                | khi chốt kết quả cuối cùng                 |

### Q7a. Cơ chế hoạt động theo từng thành phần trong sơ đồ là gì?

Để hiểu đúng hình minh họa, cần đọc theo từng thành phần con thay vì chỉ đọc tên block. Bảng dưới đây diễn giải các hộp con xuất hiện trong sơ đồ và cơ chế hoạt động của từng hộp.

| Block | Thành phần trong hình            | Cơ chế hoạt động                                                                       | Đầu ra trung gian              |
| ----- | ----------------------------------- | ------------------------------------------------------------------------------------------- | -------------------------------- |
| 0     | Parse Graphs                        | Đọc graph DOT, node/edge và nhãn topology để tạo cấu trúc đồ thị ban đầu      | graph đã parse                 |
| 0     | Align Room Tiles                    | Khớp room coordinate với node graph, snap các anchor về tile hợp lệ                   | room đã căn chỉnh với graph |
| I     | Target curve                        | Xác định ràng buộc mục tiêu cho topology hoặc grammar của dungeon                  | mục tiêu thiết kế            |
| I     | Init Pop (Grammar Genomes)          | Khởi tạo quần thể cấu trúc graph ứng viên                                           | quần thể ban đầu             |
| I     | Genetic Ops (Mutate/Cross)          | Biến đổi và lai ghép các genome để khám phá topology mới                         | topology ứng viên mới         |
| I     | Fitness (Curve match + Solve)       | Chấm điểm topology theo độ khớp đường cong mục tiêu và khả năng giải được | fitness score                    |
| I     | Graph G (Phenotype)                 | Chuyển genome đã chọn sang mission graph cụ thể                                       | mission graph                    |
| II    | 16x11 room                          | Semantic grid đầu vào của từng room                                                    | tensor room                      |
| II    | CNN Enc                             | Nén room không gian thành latent liên tục                                              | latent encoder output            |
| II    | Codebook                            | Lượng tử hóa latent thành mã rời rạc                                                | latent z rời rạc               |
| II    | Latent z                            | Biểu diễn nén để đưa sang diffusion                                                  | token/latent room                |
| II    | Optional reference-room context     | Cung cấp ngữ cảnh tham chiếu từ phòng lân cận nếu bật                             | context phụ trợ                |
| III   | Stream A: Local context             | Mã hóa lân cận N/S/E/W và ràng buộc biên                                            | local feature                    |
| III   | Local CNN                           | Trích xuất đặc trưng không gian từ các phòng lân cận                             | c_local trước fusion           |
| III   | Stream B: Global mission context    | Mã hóa graph toàn cục, loại cạnh và quan hệ nhiệm vụ                              | global feature                   |
| III   | GNN encoder (GraphGPS)              | Tóm lược topology và tín hiệu mission graph thành biểu diễn toàn cục             | c_global                         |
| III   | CrossAttention fusion               | Ghép local/global context bằng attention + residual + FFN                                 | fused context c                  |
| IV    | z_t                                 | Latent nhiễu tại bước denoise hiện tại                                                | đầu vào diffusion             |
| IV    | Conv init                           | Chiếu latent sang không gian đặc trưng của U-Net                                      | feature map khởi tạo           |
| IV    | Time Embedding                      | Mã hóa bước thời gian bằng sinusoidal + MLP                                           | t_emb                            |
| IV    | Encoder level                       | Down path của U-Net, trích xuất đặc trưng đa tỉ lệ                                 | hidden states                    |
| IV    | ResBlock                            | Cập nhật đặc trưng có điều kiện theo timestep                                      | residual features                |
| IV    | SelfAttention                       | Cho các vị trí trong cùng latent map tương tác với nhau                             | feature map đã tự chú ý     |
| IV    | CrossAttention                      | Cho latent truy vấn context từ Block III                                                  | context-aware features           |
| IV    | Bottleneck                          | Xử lý vùng đặc trưng trung tâm của U-Net                                            | latent bottleneck state          |
| IV    | Decoder level                       | Up path, phục hồi spatial detail                                                          | decoded features                 |
| IV    | SpatialGraphConditioner             | Trộn topology map và graph-grid alignment vào feature map                                | topology-aware features          |
| IV    | Conv out                            | Đưa feature map về dự đoán noise hoặc latent kế tiếp                               | z_(t-1) hoặc noise prediction   |
| IV    | Fast Sampler / Masked-Room branch   | Nhánh suy luận nhanh hoặc nhánh phòng bị che/masking tùy cấu hình                  | đường suy luận thay thế     |
| V     | LogicNet internal routing           | Định tuyến qua mạng logic để tính ràng buộc ngữ nghĩa                            | logic signal                     |
| V     | Graph (Bellman-Ford)                | Kiểm tra ràng buộc đường đi và chi phí trên graph                                 | graph-distance signal            |
| V     | Grid (CNN)                          | Đọc tín hiệu logic trên grid không gian                                               | grid logic feature               |
| V     | ∇_z log p                          | Tính gradient của xác suất logic theo latent                                            | hướng hiệu chỉnh             |
| V     | LogicNet + CFG                      | Hợp nhất score logic với classifier-free guidance                                        | guidance score                   |
| V     | Guided z*                           | Latent đã được điều hướng mềm trước khi decode                                  | latent có dẫn hướng          |
| VI    | VQ-decoder                          | Giải mã latent thành grid semantic thô                                                  | raw grid                         |
| VI    | Constrained decode                  | Giải mã có ràng buộc để giữ các vị trí quan trọng                               | raw grid có ràng buộc         |
| VI    | Marker overlay                      | Gắn lại marker hoặc nhãn ngữ nghĩa lên room                                          | grid đã gắn marker            |
| VI    | WFC repair                          | Chạy Wave Function Collapse để vá vùng còn vô nghiệm                                | room đã sửa                   |
| VI    | Stitched dungeon                    | Ghép các room đã sửa thành dungeon hoàn chỉnh                                       | dungeon đã vá                 |
| VII   | External validator                  | Kiểm tra solvability và hợp lệ bằng validator ngoài                                   | verdict kiểm tra                |
| VII   | A* pathfinding                      | Tìm đường start-goal trên grid hoặc graph                                             | kết quả connectivity           |
| VII   | Mechanical contract                 | Kiểm tra các ràng buộc cơ học và cấu trúc                                          | contract result                  |
| VII   | P-CBS behavioral probe              | Thăm dò hành vi và đặc tính chơi được                                            | behavioral probe result          |
| VII   | MAP-Elites archive                  | Ghi kết quả vào không gian đa dạng hóa chất lượng                                 | archive / population map         |
| VII   | Playable artifacts & metric reports | Tổng hợp artefact chơi được và báo cáo metric                                      | report cuối                     |

Màu sắc và kiểu mũi tên trong sơ đồ chỉ dùng để phân loại loại module và loại luồng dữ liệu. Mũi tên liên tục thể hiện luồng chạy chính, mũi tên đứt màu xanh thể hiện latent representation, còn mũi tên đứt màu nâu/đỏ thể hiện luồng conditioning hoặc phản hồi symbolic.

### Q8. Block 0 làm gì với dữ liệu đầu vào?

Block 0 là tầng chuyển dữ liệu thô sang biểu diễn có thể học được. Nó tiếp nhận dungeon gốc, đọc map lớn, graph DOT và bảng nhãn, sau đó ghép từng room với node graph tương ứng. Kết quả là một sample gồm semantic grid của room và một gói conditioning đi kèm, chẳng hạn `node_features`, `edge_index`, `edge_features`, `tpe`, `room_topology_map`, `boundary_constraints`, `neighbor_maps` và các cờ ngữ nghĩa khác.

Vai trò của Block 0 không phải là sinh nội dung mới mà là bảo đảm dữ liệu đầu vào có hình dạng, ngữ nghĩa và liên kết graph-room nhất quán. Nếu bước này sai lệch, các block phía sau sẽ học không đúng quan hệ topology.

### Q9. Block I là gì trong sơ đồ này?

Block I là tầng tạo hoặc điều phối mission graph theo hướng search-based. Ở mức mô hình hóa, block này xác định cấu trúc nhiệm vụ của dungeon, bao gồm phòng khởi đầu, phòng khóa, phòng boss, phòng mục tiêu cuối và quan hệ đi qua giữa các phòng.

Trong kiến trúc này, search không chỉ đóng vai trò tìm đường mà còn phải bảo đảm cấu trúc dungeon có thể chơi được và có nhịp độ tiến triển hợp lý. Vì vậy, Block I được xem là tầng dựng topology trước khi sinh room ở mức neural.

### Q10. Block II làm gì với room tensor?

Block II là Semantic VQ-VAE Tokenizer. Đây là encoder phục vụ học biểu diễn room ở mức rời rạc, và cần được phân biệt rõ với bộ encoder điều kiện ở Block III.

Đầu vào của block này là semantic room tensor, ví dụ room kích thước `[1, 16, 11]` hoặc one-hot tile map `[44, 16, 11]` khi huấn luyện. Encoder CNN nén room thành latent `z_e`, vector quantizer ép latent đó về một codebook rời rạc `z_q`, rồi decoder khôi phục lại room. Về mặt biểu diễn, codebook chuyển không gian liên tục thành một “từ vựng” room rời rạc mà diffusion có thể sử dụng hiệu quả hơn.

Về mặt công thức, có thể đọc ngắn gọn như sau:

`z_e = E(x)`

`z_q = argmin_{e_k} ||z_e - e_k||_2`

`x_hat = D(z_q)`

Trong code, block này tối ưu đồng thời reconstruction loss, codebook loss và commitment loss. Điều đó buộc encoder phải nén đủ thông tin, duy trì sự ổn định của codebook và tránh tình trạng latent trôi tự do ra khỏi miền embedding đã học.

### Q11. Block III gom local và global context như thế nào?

Đây là block dễ gây nhầm lẫn về thuật ngữ encoder, vì nó không phải encoder CNN của room mà là bộ mã hóa điều kiện phục vụ diffusion.

Block III bao gồm hai luồng chính. Luồng local là `LocalStreamEncoder`, nhận latent của bốn phòng lân cận, boundary constraints và vị trí room. Bên trong, bốn nhánh mã hóa theo hướng N/S/E/W, một encoder cho ràng buộc biên, một encoder cho vị trí và một MLP fusion được sử dụng để tạo `c_local`.

Luồng global là `GlobalStreamEncoder`, nhận node features, edge index, edge features, TPE và khoảng cách tới current node. Tùy cấu hình, module này có thể sử dụng GCN, GATv2, SAGE hoặc encoder kiểu GraphGPS. Kết quả thu được là `c_global`, tức biểu diễn graph ở mức toàn cục hoặc token của node hiện tại.

Sau đó, `CrossAttentionFusion` sử dụng `c_local` làm query và `c_global` làm key/value để tạo điều kiện cuối cùng. Khi bật `style_id`, block còn chèn một style token toàn dungeon; khi bật `reference_room_maps`, nó ghép thêm exemplar map từ các phòng lân cận. Đầu ra cuối cùng thường là một vector conditioning có kích thước `[B, 256]`.

Vì vậy, Block III không sinh room trực tiếp mà đóng vai trò rút gọn ngữ cảnh thành một biểu diễn điều kiện thống nhất cho diffusion.

### Q12. Có bao nhiêu loại encoder trong mô hình?

Có ít nhất hai nghĩa khác nhau của từ “encoder”.

| Loại encoder                   | Thuộc block | Nhiệm vụ                                                                              |
| ------------------------------- | ------------ | --------------------------------------------------------------------------------------- |
| VQ-VAE encoder                  | Block II     | nén một room thành latent rời rạc cho tokenizer                                    |
| Local/Global condition encoders | Block III    | nén ngữ cảnh phòng lân cận và ngữ cảnh graph thành điều kiện cho diffusion |

Nếu đọc sơ đồ mà không tách hai lớp này, rất dễ hiểu nhầm rằng mô hình chỉ có một encoder duy nhất. Thực tế, Block II là encoder của chính dữ liệu room, còn Block III là encoder của điều kiện sinh.

### Q13. Block IV khác gì với một U-Net thường?

Block IV là Latent Diffusion trên latent của VQ-VAE, nhưng U-Net ở đây không hoạt động độc lập. Nó được điều kiện hóa bởi Block III, bởi topology map và, trong một số trường hợp, bởi graph nodes ở mức vị trí cụ thể.

Luồng xử lý chuẩn là: noisy latent `x_t` đi qua `input_proj`, sau đó qua nhiều `DownBlock`, `mid block` và `UpBlock`. Mỗi `ResBlock` nhận timestep embedding để xác định bước denoise hiện tại. Mỗi `AttentionBlock` thực hiện đồng thời self-attention trên grid và cross-attention với context. Cuối cùng, `output_proj` trả về dự đoán noise hoặc v-prediction tùy theo cấu hình.

Điểm quan trọng là diffusion không chỉ khử nhiễu theo nghĩa thị giác mà còn khử nhiễu trong không gian latent của room, đồng thời bảo toàn các ràng buộc topology và ngữ nghĩa của dungeon.

### Q14. Diffusion block có những thành phần nội bộ nào?

Các thành phần chính của Block IV gồm:

- `TimestepEmbedding`: biến chỉ số thời gian denoise thành vector điều kiện.
- `ResBlock`: khối residual có scale-shift theo timestep.
- `SelfAttention`: cho các vị trí grid nhìn nhau trong cùng latent map.
- `CrossAttention`: cho grid nhìn sang context token.
- `SpatialGraphConditioner`: ghép `room_topology_map` và graph nodes vào grid feature.
- `GraphToGridCrossAttention`: cho từng ô grid hỏi trực tiếp các node graph.
- `RoomTopologyConditioner`: đưa topology map vào U-Net theo kiểu additive hoặc SPADE.
- `GradientGuidance`: áp dụng logic loss gradient khi sampling.

Về mặt hệ thống, kiến trúc này cho phép mỗi vị trí trong latent không chỉ chịu tác động của nhiễu hiện tại mà còn được điều kiện hóa bởi context, graph và topology prior của room.

### Q15. Cross-attention hoạt động bên trong thế nào?

Có hai lớp cross-attention khác nhau trong hệ thống, và cả hai đều quan trọng.

Lớp thứ nhất nằm ở Block III. Tại đây, một vector local summary đóng vai trò query, còn các token graph đóng vai trò key/value. Mục tiêu của cơ chế này là rút ra một conditioning vector chung cho toàn bộ room.

Lớp thứ hai nằm ở Block IV. Tại đây, mỗi vị trí trên grid latent là query, còn context tokens hoặc graph nodes là key/value. Có thể hiểu đây là cơ chế để từng vị trí lưới truy vấn thông tin từ graph.

Về mặt cơ chế, công thức cơ bản là:

`Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V`

Trong `CrossAttention` của diffusion, context còn có thể được refine bằng topology-aware message passing trước khi tạo K/V. Điều này có nghĩa là graph token không đi trực tiếp vào attention theo dạng thô, mà được làm giàu bằng quan hệ cạnh trước.

Trong `GraphToGridCrossAttention`, grid features được flatten thành sequence, graph nodes được chiếu qua `graph_pe`, độ bậc vào/ra của node được biến thành bias cấu trúc, rồi Q/K/V mới được tạo. Khi số node vượt ngưỡng, module có thể chuyển sang `linear_hedgehog` để giảm chi phí tính toán. Nếu topology map đã có sẵn, `SpatialGraphConditioner` sẽ hợp nhất thêm thông tin này thông qua gate học được.

### Q16. Block V làm gì trong lúc sinh mẫu?

Block V là cơ chế logic steering, được thiết kế để bảo đảm mẫu sinh tuân thủ các ràng buộc logic chứ không chỉ phù hợp về mặt phân phối thống kê.

Trong code, `GradientGuidance` nhận `x_t` hiện tại và một gói `graph_data`. Nó lọc và giới hạn dữ liệu graph để tránh payload quá lớn, sau đó gọi `logic_net` để thu được logic loss. Từ loss này, module tính gradient theo `x_t` và trừ gradient đó khỏi mean dự đoán của diffusion.

`x_hat_{t-1} = mu_theta(x_t, t, c) - gamma * grad_x_t L_logic`

Nếu `logic_net` không được cấu hình, block này trả về zero guidance. Khi gradient có biên độ quá lớn, nó được giới hạn theo norm để tránh làm mất ổn định quá trình sampling. Vì vậy, Block V nên được hiểu như một tầng hiệu chỉnh theo ràng buộc logic hơn là một khối sinh nội dung độc lập.

### Q17. Block VI có vai trò gì nếu model neural đã sinh ra mẫu?

Block VI là tầng sửa lỗi cuối cùng, tức giai đoạn chuyển từ sinh mẫu sang khôi phục mẫu.

`PathAnalyzer` trước hết kiểm tra đường đi bằng A* trên grid hoặc kiểm tra connectivity trên graph. Khi phát hiện đoạn bị chặn, thiếu key hoặc disconnected, `EntropyReset` sẽ tạo mask quanh vùng lỗi. Sau đó `WaveFunctionCollapse` khởi tạo phân phối xác suất trên từng cell, luôn chọn cell có entropy thấp nhất, collapse ô đó về một tile cụ thể và propagate ràng buộc tương thích sang các ô lân cận.

Sau WFC, `ConstraintPropagator` có thể khôi phục lại connectivity giữa start và goal bằng BFS hoặc bằng một đường đi đơn giản nếu cần. `SymbolicRefiner` điều phối toàn bộ vòng lặp này, đồng thời ghi nhận diagnostics và có thể gọi feedback neural nếu WFC gặp bế tắc.

Vì vậy, block này đóng vai trò một cơ chế hiệu chỉnh hậu nghiệm, bảo đảm kết quả sinh không rơi vào trạng thái vô nghiệm.

### Q18. Block VII đo cái gì?

Block VII không sinh room mới mà thực hiện đánh giá đầu ra cuối cùng.

Tầng này thường kiểm tra các tiêu chí như solvability, độ liên thông, tính hợp lệ của đường đi, mức độ đa dạng và các chỉ số quality-diversity nếu pipeline đang chạy theo hướng QD evaluation. Do đó, Block VII là khâu tổng kết cuối cùng để xác định dungeon có đạt yêu cầu hay không.

### Q19. Bộ tensor quan trọng nhất là gì?

Bảng dưới đây tóm tắt các shape thường gặp nhất khi đọc đúng luồng của mô hình.

| Thành phần             | Shape ví dụ                         | Ý nghĩa                         |
| ------------------------ | ------------------------------------- | --------------------------------- |
| room tensor              | `[1, 16, 11]`                       | semantic grid của một room      |
| one-hot room             | `[44, 16, 11]`                      | đầu vào training của VQ-VAE   |
| `z_e`                  | `[64, 4, 3]`                        | latent trước quantization       |
| `z_q`                  | `[64, 4, 3]`                        | latent sau quantization           |
| `indices`              | `[4, 3]`                            | chỉ số codebook rời rạc       |
| `c_local`              | `[B, 256]`                          | điều kiện từ phòng lân cận |
| `c_global`             | `[B, N, 256]` hoặc `[B, 1, 256]` | điều kiện từ graph            |
| fused condition          | `[B, 256]`                          | đầu ra của Block III           |
| noisy latent             | `[B, 64, 4, 3]`                     | đầu vào của diffusion         |
| predicted noise          | `[B, 64, 4, 3]`                     | đầu ra của diffusion U-Net     |
| `room_topology_map`    | `[54, 16, 11]`                      | topology supervision map          |
| `boundary_constraints` | `[8]`                               | ràng buộc biên cho phòng      |
| `neighbor_maps`        | `dict(4)`                           | map của bốn phòng lân cận    |

### Q20. Dùng pipeline này như thế nào trong thực tế?

Nếu xét theo trình tự thực thi, quy trình hợp lý nhất là:

1. Nạp dungeon gốc và ghép room với graph.
2. Dùng Block II để học representation rời rạc của từng room.
3. Dùng Block III để sinh điều kiện từ room lân cận và mission graph.
4. Dùng Block IV để sinh latent room dưới ràng buộc topology.
5. Dùng Block V nếu muốn ép mẫu hợp luật hơn trong sampling.
6. Dùng Block VI nếu validator báo room hoặc dungeon còn lỗi.
7. Dùng Block VII để tổng kết độ hợp lệ và chất lượng cuối.

Theo cách tiếp cận này, mô hình không phải một mạng sinh đơn lẻ mà là một chuỗi từ biểu diễn đến kiểm tra, trong đó mỗi block đảm nhiệm một chức năng rõ ràng.

## 5) Kết luận ngắn

Với corpus Zelda này, khía cạnh cốt lõi không nằm ở từng map tile riêng lẻ mà ở cơ chế parse graph và ánh xạ ngược về semantic grid của từng room. Trên cơ sở đó, pipeline học biểu diễn latent cho room, điều kiện hóa bằng mission graph, sinh room bằng diffusion và hiệu chỉnh lỗi bằng logic cùng symbolic repair.

Nếu cần mở rộng tài liệu, phần hợp lý tiếp theo là một walkthrough tuyến tính theo đúng trình tự thực thi: load dungeon -> match graph -> build room sample -> train VQ-VAE -> train diffusion -> sample -> repair -> validate.

## 6) Component-level: cơ chế hoạt động chi tiết cho từng thành phần (khối và hộp con)

Phần này bổ sung mô tả cơ chế hoạt động nội bộ cho từng hộp con chính xuất hiện trên sơ đồ, nêu rõ file/cấu trúc triển khai tham chiếu, đầu vào, bước xử lý chính và đầu ra trung gian. Mục tiêu là để người đọc nhìn vào sơ đồ và biết mỗi hộp "bên trong chạy gì".

### Block 0 — Data Adapter (parse / align)

- Triển khai chính: loader và adapter trong `src/zelda_data/zelda_loader.py` và `src/zelda_data/zelda_core.py`.
- Đầu vào: raw map files, DOT graph, label map.
- Bước xử lý: parse DOT → tạo `node_features`, `edge_index`, `edge_attr` → match rooms ↔ graph bằng BFS / nearest-walkable heuristics (`match_rooms_to_graph_impl`) → snap anchors (start/door/goal) về tile có thể đi được → xây `room_topology_map`, `boundary_constraints`, `neighbor_maps`.
- Đầu ra: tuple `(grid_tensor, graph_dict)` với các trường đã nêu (shapes trong phần ví dụ). Các giá trị này là nguồn sự thật để tạo supervision cho Block II–IV.

### Block I — Search-based Mission Graph Generation (chi tiết kỹ thuật)

Mục tiêu: sinh một mission graph khả thi (phenotype) phù hợp với mục tiêu thiết kế (target curve, pacing, difficulty) và các ràng buộc chơi được (solvability, key/lock consistency). Graph này là điều kiện global đầu vào cho Block III.

1) Tổng quan và đầu vào/đầu ra

- Đầu vào chính: mục tiêu thiết kế (target curve hoặc tập đặc trưng mong muốn), ràng buộc (số phòng tối thiểu/tối đa, yêu cầu boss/key/lock), seed ngẫu nhiên và hyperparameters tìm kiếm.
- Đầu ra: Mission graph G = (V, E, y_V, y_E) với tập node V, edge E, và nhãn node/edge (ví dụ: start, goal, key, lock, transition-type).

2) Biểu diễn (genotype ↔ phenotype)

- Genotype: biểu diễn rời rạc thuận tiện cho thao tác search (ví dụ adjacency matrix compressed, list of node specs, hoặc một chuỗi production từ grammar). Mỗi cá thể (genome) chứa thông tin: node types, node order (optional), edge list, và tham số meta (e.g., weight cho subgraph patterns).
- Phenotype: một đồ thị có thể kiểm tra được (node/edge labels) dùng để đánh giá fitness; chuyển đổi genotype→phenotype là bước decode (apply productions hoặc xây adjacency).

3) Phép sinh bằng grammar (grammar-based generation)

- Nếu dùng grammar, định nghĩa một graph grammar G_{gram} = (N, Σ, P, S) trong đó:
  - N: nonterminals (kinds of partial structures, e.g., corridor, hub)
  - Σ: terminals (final node types)
  - P: production rules (ví dụ R: hub -> hub + corridor + room)
  - S: start symbol (sơ đồ trừu tượng ban đầu)
- Generation: lặp áp dụng các production lên nonterminals theo thứ tự/heuristic để mở rộng tới trạng thái chỉ chứa terminals. Grammar giúp đảm bảo một số invariant (ví dụ tồn tại boss node duy nhất, chain length bounds).

4) Thuật toán tìm kiếm: Genetic Algorithm (ví dụ) và biến thể

- Cơ chế phổ biến trong repo: GA-like population search (có thể kết hợp hill-climbing, beam search hoặc simulated annealing).
- Thành phần:
  - Khởi tạo: tạo population P_0 gồm M genome bằng heuristic (random + seeding từ templates).
  - Selection: chọn theo tournament hoặc roulette-wheel theo score.
  - Crossover: nối ghép hai genome (edge/node splice, subtree exchange đối với grammar representations).
  - Mutation: thêm/xóa node, thêm/xóa edge, đổi nhãn node, perturb metadata (vị trí boss, difficulty weight).
  - Replacement: tạo population tiếp theo P_{t+1} bằng elitism + offspring.

Pseudocode (GA simplified):

```
Initialize P ← {g_i}_{i=1..M}
for gen in 1..G:
	Evaluate fitness for each g ∈ P
	P_elite ← top-k(P)
	Offspring ← {}
	while |Offspring| < M - k:
		a, b ← select(P)
		c ← crossover(a,b)
		c' ← mutate(c)
		Offspring.add(c')
	P ← P_elite ∪ Offspring
return best g ∈ P
```

5) Hàm fitness — công thức tổng quát
   Fitness được thiết kế để cân bằng độ tương thích với target curve, khả năng giải (solvability) và hình phạt các vi phạm ràng buộc. Một dạng chung:

$$
Fitness(G) \,=\, \lambda_{curve} \; S_{curve}(G) \; + \; \lambda_{solve} \; S_{solve}(G) \; - \; \lambda_{pen} \; Pen(G)
$$

Trong đó:

- $S_{curve}(G)$: similarity score giữa phân bố đặc trưng graph $\phi(G)$ và target curve $T$ — ví dụ $S_{curve} = -\|\phi(G)-T\|_2$ hoặc là hệ số tương quan cosine.
- $S_{solve}(G)$: solvability score (binay/soft) biểu diễn khả năng tìm đường start→goal, tồn tại key/lock path, hay metric đường đi mong muốn; thường lấy giá trị trong $[0,1]$ (1 nếu thoả điều kiện). Có thể dùng một hàm giảm dần theo chiều dài đường đi: $S_{solve} = \exp(-\alpha (L_{path} - L_{target})^2)$.
- $Pen(G)$: tổng các phạt theo ràng buộc (ví dụ cặp key/lock không cân bằng, vượt limits node count, disconnected components).

Ví dụ cụ thể (công thức):

$$
S_{curve}(G) = -\|\mathrm{deg\_hist}(G) - T\|_2
$$

$$
S_{solve}(G) = \begin{cases} 1 &\text{if A^*(G) finds valid path satisfying constraints}\\ 0 &\text{otherwise} \end{cases}
$$

Tổng hợp:

$$
Fitness(G) = \lambda_{curve} \left(-\|\mathrm{deg\_hist}(G)-T\|_2\right) + \lambda_{solve} \, S_{solve}(G) - \lambda_{pen} \, Pen(G)
$$

6) Kiểm tra solvability (A* / path-based probes)

- Mỗi candidate G cần được kiểm tra bằng solver nhẹ:
  - Chuyển graph nodes → grid proxies (nếu cần) hoặc dùng trực tiếp graph (node adjacency)
  - Chạy A* / BFS/shortest-path từ start node đến goal node, với cost metric (có thể bao gồm door/locked costs) để xác minh tính khả thi.
- Nếu triển khai là mission-graph-only (không embed vào grid), dùng graph-level path find; nếu cần kiểm tra tile-level tương ứng, decode node→room footprint và chạy A* trên grid tái tạo.

7) Heuristics và practical choices

- Population seeding: bắt đầu với một số cấu trúc template (linear chain, hub-and-spoke) giúp hội tụ nhanh hơn.
- Fitness shaping: sử dụng soft penalties và annealing weights (tăng $\lambda_{solve}$ khi tiến gần cuối generations) để đảm bảo đầu ra vừa đa dạng vừa hợp lệ.
- Budgeting: giới hạn phép tính solver (A*) bằng timeout hoặc early-abort để giữ tốc độ tìm kiếm.
- Hybrid search: có thể kết hợp beam search trên không gian grammar để nhanh tạo ra các khuôn mẫu, sau đó chạy GA tinh chỉnh chi tiết.

8) Output và tích hợp vào pipeline

- Khi chọn xong graph G^*, module xuất graph dưới dạng dict (node_features, edge_index, edge_attr, tpe, start_node_id, etc.) — cùng định dạng với loader để Block III có thể trực tiếp mã hóa nó (ví dụ `fused condition c` sử dụng node tokens từ G^*).
- Lưu ý: mission graph có thể được giữ cố định (single-shot generation) hoặc được đa biến (generate ensemble) để sampling đa dạng trong Block IV.

### Block II — Semantic VQ-VAE Tokenizer

Block II là tầng nén biểu diễn room sang latent rời rạc. Mục tiêu của block này không phải sinh phòng mới ngay lập tức mà là học một không gian codebook đủ giàu để giữ lại cấu trúc ngữ nghĩa của room nhưng vẫn đủ rời rạc để diffusion có thể sinh hiệu quả hơn.

- Triển khai chính: `src/core/vqvae.py` với encoder `E`, decoder `D` và `VectorQuantizer`/codebook.
- Đầu vào: semantic room tensor `[C, H, W]` như `[44, 16, 11]` (one-hot tile map) hoặc room tensor đã normalize.
- Bước xử lý nội bộ:
  - `E(x)` dùng CNN/conv blocks để trích đặc trưng không gian và giảm chiều từ tile-space sang latent-space.
  - Latent liên tục `z_e` được đưa vào codebook, chọn vector gần nhất theo khoảng cách Euclid để thu được `z_q`.
  - `D(z_q)` giải mã latent rời rạc trở lại room reconstruction `x_hat`.
- Đầu ra: `z_e`, `z_q`, chỉ số codebook `indices`, và reconstructed room `x_hat`.

Về công thức, có thể mô tả ngắn gọn như sau:

$$
z_e = E(x)
$$

$$
q(x) = \arg\min_{e_k \in \mathcal{E}} \| z_e - e_k \|_2
$$

$$
z_q = e_{q(x)}, \qquad x_{hat} = D(z_q)
$$

Trong đó `\mathcal{E}` là tập vector trong codebook. Hàm mất mát thường gồm ba phần:

$$
\mathcal{L}_{VQ-VAE} = \mathcal{L}_{rec}(x, x_{hat}) + \mathcal{L}_{codebook} + \beta \, \mathcal{L}_{commit}
$$

- `\mathcal{L}_{rec}`: reconstruction loss, thường là cross-entropy hoặc L1/L2 tùy encoding của tile.
- `\mathcal{L}_{codebook}`: kéo codebook vector tiến gần latent encoder output.
- `\mathcal{L}_{commit}`: buộc encoder cam kết đi theo codebook đã học, tránh latent trôi tự do.

Diễn giải cơ chế: encoder không cố học một biểu diễn liên tục rất mịn như autoencoder thường, mà học một hệ từ điển hữu hạn. Điều này làm latent của room trở thành chuỗi/chỉ số rời rạc, thuận lợi cho diffusion, sampling và phân phối thống kê ổn định hơn.

### Block III — Dual-Stream Conditioning

Block III là tầng gom ngữ cảnh, trong đó một luồng đọc thông tin cục bộ của room và một luồng đọc mission graph toàn cục. Hai luồng này được ghép lại thành một conditioning vector dùng cho diffusion.

- Triển khai chính: `src/core/condition_encoder.py` với `LocalStreamEncoder`, `GlobalStreamEncoder` và `CrossAttentionFusion`.
- Đầu vào: local neighbor maps, boundary constraints, vị trí room, graph tokens (`node_features`, `edge_index`, `edge_attr`, `tpe`) và tùy chọn reference room maps.
- Bước xử lý nội bộ:
  - Local stream mã hóa các vùng lân cận theo N/S/E/W để nắm cấu trúc biên, hành lang và vị trí tương đối.
  - Global stream chạy GNN/GraphGPS để mã hóa topology, edge semantics và node roles của mission graph.
  - Fusion block dùng cross-attention và residual connections để trộn local/global context thành vector thống nhất.
- Đầu ra: `c_local`, `c_global`, và fused condition `c`.

Luồng local thường được hiểu như một encoder ngữ cảnh không gian gần. Nếu ký hiệu các nhánh là `f_N, f_S, f_E, f_W` và các encoder ràng buộc biên là `f_b`, encoder vị trí là `f_p`, thì một dạng gom đơn giản là:

$$
c_{local} = \mathrm{MLP}_{fuse}([f_N; f_S; f_E; f_W; f_b; f_p])
$$

Luồng global có thể mô tả bằng message passing trên graph:

$$
h_v^{(l+1)} = \phi\Big(h_v^{(l)}, \; \square_{u \in \mathcal{N}(v)} \psi(h_u^{(l)}, e_{uv})\Big)
$$

Trong đó `\phi` và `\psi` là các hàm học được, còn `\square` là phép gộp như sum/mean/attention. Kết quả cuối là `c_global`, chứa topology-level mission information.

Sau đó cross-attention fusion hoạt động theo công thức chuẩn:

$$
\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\Big(\frac{QK^T}{\sqrt{d}}\Big)V
$$

Trong Block III, `c_local` thường đóng vai trò query để hỏi `c_global` xem room hiện tại đang nằm trong ngữ cảnh nhiệm vụ nào. Điều này giúp model biết room phải “ứng xử” ra sao thay vì chỉ nhìn tile-local pattern.

### Block IV — Latent Diffusion (U-Net + attention stack)

Block IV là lõi sinh mẫu. Tại đây latent room nhiễu `z_t` được khử nhiễu từng bước dưới ảnh hưởng của timestep embedding, cross-attention với conditioning vector, topology map và graph-aware modules.

- Triển khai chính: `src/core/latent_diffusion.py` và các module topology-graph trong `src/core/graph_grid_attention.py`.
- Đầu vào: noisy latent `z_t`, timestep `t`, conditioning `c`, `room_topology_map`, và graph tokens.
- Bước xử lý nội bộ:
  - `Conv init` chiếu latent vào kênh ẩn của U-Net.
  - Down path trích feature đa tỉ lệ bằng `ResBlock` và `SelfAttention`.
  - Mid block gom ngữ cảnh dài hạn.
  - Up path phục hồi độ phân giải bằng skip connection, `CrossAttention` và các conditioning modules.
  - `Conv out` sinh ra noise prediction hoặc `v`-prediction.
- Đầu ra: `\hat\epsilon` hoặc `\hat v`, rồi suy ra `z_{t-1}`.

Một bước denoise chuẩn có thể viết theo dạng:

$$
z_{t-1} = \mu_{\theta}(z_t, t, c) + \sigma_t \cdot \eta
$$

với `\eta \sim \mathcal{N}(0, I)`. Nếu model dùng noise prediction thì:

$$
\mathcal{L}_{diff} = \| \epsilon - \epsilon_{\theta}(z_t, t, c) \|_2^2
$$

Nếu dùng `v_prediction`, mục tiêu học sẽ được đổi theo parametrization tương ứng nhưng ý nghĩa vẫn là khử nhiễu có điều kiện.

Về nội bộ attention stack, U-Net không chỉ tự chú ý trên grid mà còn được điều kiện hóa theo graph:

- `SelfAttention` cho phép mỗi vị trí latent nhìn các vị trí khác trong cùng map.
- `CrossAttention` cho phép latent truy vấn `c` hoặc graph tokens.
- `SpatialGraphConditioner` chèn topology prior từ `room_topology_map` vào feature map.
- `GraphToGridCrossAttention` cho phép từng ô grid hỏi node graph trực tiếp.
- `RoomTopologyConditioner` đưa topology vào feature modulation hoặc attention bias.

Điểm quan trọng là Block IV hoạt động ở latent-space chứ không ở pixel-space. Nghĩa là model học cấu trúc room trong không gian nén, giúp sampling nhanh hơn và dễ gắn với topology hơn.

### Block V — LogicNet steering (gradient guidance)

Block V là tầng dẫn hướng bằng logic gradient. Nó không sinh ra một room độc lập mà điều chỉnh kết quả sinh của diffusion để giảm vi phạm logic.

- Triển khai chính: logic guidance module tích hợp với sampler trong `src/core/latent_diffusion.py` hoặc module logic riêng.
- Đầu vào: latent hiện tại `z_t`, graph/mission data, và thông tin chỉ dẫn điều kiện.
- Bước xử lý nội bộ:
  - Tính một logic score hoặc logic loss từ `logic_net`.
  - Lấy gradient của loss theo latent.
  - Trừ gradient này vào update của diffusion sampler để ép mẫu hợp luật hơn.
- Đầu ra: guided latent `z^*`.

Một cách viết khái quát:

$$
\mathcal{L}_{logic} = \mathcal{L}_{path} + \mathcal{L}_{key/lock} + \mathcal{L}_{connectivity} + \mathcal{L}_{role}
$$

$$
g = \nabla_{z_t} \mathcal{L}_{logic}
$$

$$
z_{t-1}^{guided} = z_{t-1}^{diffusion} - \gamma \cdot \mathrm{clip}(g)
$$

Trong đó:

- `\mathcal{L}_{path}` đo đường đi từ start đến goal có hợp lệ không.
- `\mathcal{L}_{key/lock}` đo key/lock có xuất hiện đúng thứ tự không.
- `\mathcal{L}_{connectivity}` đo kết nối tổng thể của graph hoặc room graph.
- `\mathcal{L}_{role}` đo xem node/room có giữ đúng vai trò nhiệm vụ hay không.

Nếu `logic_net` không bật, block này trả về guidance bằng 0 và diffusion chạy như bình thường. Nếu gradient quá lớn, hệ thống clip norm để tránh phá vỡ quá trình sampling.

### Block VI — Symbolic repair & overlay

Block VI là tầng sửa lỗi symbolic sau khi neural model sinh xong. Nó dùng các phép kiểm tra cấu trúc, constraint propagation và WFC để vá những vùng chưa hợp lệ.

- Triển khai chính: `src/core/symbolic_refiner.py` với `PathAnalyzer`, `EntropyReset`, `WaveFunctionCollapse`, `ConstraintPropagator`, `SymbolicRefiner`.
- Đầu vào: raw decoded grid, marker overlay, ràng buộc topology và các diagnostics từ validator.
- Bước xử lý nội bộ:
  - Kiểm tra đường đi và connectivity.
  - Xác định vùng lỗi và reset entropy vùng đó.
  - Chạy WFC để collapse tile hợp lệ nhất trước, rồi propagate constraints sang các ô lân cận.
  - Nếu cần, overlay lại marker và các tile đặc biệt để giữ vai trò nhiệm vụ.
- Đầu ra: repaired room grid hoặc stitched dungeon đã được vá.

WFC trong block này có thể diễn giải bằng chu trình:

1. Tính entropy cho từng ô chưa cố định.
2. Chọn ô có entropy nhỏ nhất.
3. Collapse ô đó về một tile hợp lệ theo distribution.
4. Propagate các ràng buộc tương thích sang ô xung quanh.
5. Lặp đến khi hết ô chưa xác định hoặc gặp mâu thuẫn.

Nếu gọi `X` là tập ô chưa collapse, thì tại mỗi bước chọn:

$$
i^* = \arg\min_{i \in X} H(i)
$$

với `H(i)` là entropy của ô `i`. Đây là lý do WFC thường sửa được các khu vực local defect mà neural sampling chưa đảm bảo.

### Block VII — Validation & QD evaluation

Block VII là khâu đánh giá cuối cùng. Nó không tạo mẫu mới mà đo xem dungeon sinh ra có chơi được, ổn định và đa dạng hay không.

- Triển khai chính: `ExternalValidator`, `SolvabilityChecker`, `StateSpaceAStar`, contract scorer trong `src/pipeline/dungeon_pipeline.py`, và `P-CBS` trong `src/evaluation/pcbs_validation.py` / `src/simulation/cognitive_bounded_search.py`.
- Đầu vào: dungeon đã stitch, configuration đánh giá, persona settings, và metric set.
- Bước xử lý nội bộ:
  - Hard oracle kiểm tra solvability trên semantic grid đã chuẩn hóa.
  - A* / BFS / pathfinding probes xác nhận đường đi và connectivity.
  - Mechanical contract probe đo xem phòng/dungeon có “đọc được” về mặt cấu trúc cơ học hay không.
  - Behavioral probe bằng `P-CBS` mô phỏng nhiều persona người chơi để đo trải nghiệm và độ chịu đựng của layout.
  - Ghi kết quả vào QD archive nếu đang tối ưu đa mục tiêu.
- Đầu ra: metrics, verdict, archive entry, playable artifacts.

Nếu xét theo quality-diversity, một feature vector của mẫu có thể là:

$$
f(G) = [f_1(G), f_2(G), \dots, f_m(G)]
$$

trong đó các chiều có thể là độ tuyến tính của layout, độ leniency, số nhánh, chiều dài đường đi, hoặc độ phức tạp logic. MAP-Elites sẽ đặt mẫu vào ô archive tương ứng với feature descriptor và giữ cá thể tốt nhất trong từng ô.

Với validator ngoài, đầu ra thường được dùng để phân loại mẫu theo các trạng thái như: valid / solvable / partially valid / invalid. Điều này cho phép pipeline có feedback rõ ràng về việc diffusion và symbolic repair đã đạt đến mức nào.

### Q18a. Mechanical contract probe là gì?

Mechanical contract probe là bộ kiểm tra xem dungeon có “đúng cơ học” hay không, tức các thành phần chức năng của room có tạo thành một hợp đồng chơi được rõ ràng giữa start, đường đi chính, tương tác trạng thái và mục tiêu hay không. Đây không chỉ là câu hỏi “có đường đi tới đích không”, mà còn là “đường đi đó có được đóng khung bằng cấu trúc đủ đọc được để người chơi hiểu và thực hiện không”.

Trong code, cơ chế này được thể hiện rõ ở hàm `_evaluate_puzzle_candidate_contract(...)` trong `src/pipeline/dungeon_pipeline.py`. Hàm này tính một contract score dựa trên các thành phần như:
- `path_exists`: đường chính có tồn tại hay không.
- `projected_stateful`: vị trí tương tác trạng thái có thể chiếu về ô đi được hay không.
- `pocket_floor_tiles`: số tile đi được trong “túi tương tác” xung quanh điểm trạng thái.
- `frame_block_tiles`: số tile tường/block tạo khung cơ học xung quanh vùng tương tác.
- `anchor_adjacent_walkable`: số ô đi được kề cận với anchor.
- `stateful_distance_to_path` và `stateful_branch_gain`: độ gần của trạng thái với đường chính và mức độ tạo nhánh hợp lý.

Mô hình scoring của contract có thể tóm tắt như sau:

$$
S_{contract} = 0.30\,\mathbf{1}[path] + 0.25\,\mathbf{1}[stateful] + 0.35\,f_{pocket} + 0.35\,f_{frame} + 0.20\,f_{adj} + 0.25\,f_{distance} + 0.25\,f_{branch} - P_{fail}
$$

trong đó `f_{pocket}`, `f_{frame}`, `f_{adj}`, `f_{distance}` và `f_{branch}` là các thành phần chuẩn hóa về miền `[0,1]`, còn `P_{fail}` là phạt nếu vi phạm các điều kiện như:
- thiếu path chính,
- thiếu anchor trạng thái,
- vùng tương tác quá yếu,
- anchor bị kín,
- anchor nằm quá xa đường chính,
- không có detour cho khóa/vật phẩm khi gate family yêu cầu.

Nói ngắn gọn, mechanical contract probe đo mức độ “có cấu trúc cơ học” của room: nó kiểm tra xem room có một lõi di chuyển rõ ràng, có điểm tương tác hợp lý, và có khung chặn/khung mở đủ để người chơi đọc vai trò của room hay không.

### Q18b. Behavioral probe là gì?

Behavioral probe là lớp kiểm tra mô phỏng cách các kiểu người chơi khác nhau giải và cảm nhận dungeon. Thay vì chỉ dùng một bộ giải tối ưu kiểu hard oracle, hệ thống dùng `P-CBS` (`Persona-Driven Cognitive Bounded Search`) để mô phỏng các persona như `Balanced`, `Explorer`, `Cautious`, `Forgetful`, `Speedrunner`, `Greedy`, `Completionist`, và `Novice`.

Thành phần này nằm trong `src/evaluation/pcbs_validation.py` và engine persona trong `src/simulation/cognitive_bounded_search.py`. Khác với A* thuần tối ưu, `P-CBS` dùng utility bị ràng buộc bởi persona, bộ nhớ hữu hạn, và các penalty cho việc vòng lại, nguy cơ, hoặc độ phức tạp cục bộ. Vì thế nó không chỉ hỏi “có giải được không”, mà còn hỏi “một người chơi với kiểu hành vi cụ thể có giải được và hiểu được dungeon này không”.

Các tín hiệu hành vi thường được tổng hợp thành:
- `persona`: loại người chơi được mô phỏng.
- `path_length` / `path_efficiency`: độ hiệu quả đường đi của persona.
- `confusion_ratio` hoặc `confusion_index`: mức độ lạc hướng / nhiễu nhận thức.
- `success` / `solvable`: persona có tìm được lời giải hay không.
- `generation_time_sec`: chi phí suy luận / mô phỏng.

Về mặt khái niệm, P-CBS sử dụng hàm mục tiêu dạng persona-conditioned utility:

$$
U_{persona}(s) = \alpha \cdot Goal(s) - \beta \cdot Risk(s) - \gamma \cdot Revisitation(s) - \delta \cdot Complexity(s) + \eta \cdot MemoryFit(s)
$$

Trong đó các hệ số `\alpha, \beta, \gamma, \delta, \eta` phụ thuộc vào persona. Ví dụ:
- `Speedrunner` ưu tiên `Goal` và `path_efficiency`.
- `Explorer` chấp nhận `Complexity` cao hơn để đổi lấy coverage.
- `Cautious` phạt `Risk` mạnh hơn.
- `Forgetful` bị ảnh hưởng bởi `MemoryFit` thấp, nên dễ lạc hướng hơn.
- `Completionist` có xu hướng theo đuổi các mục tiêu phụ và kiểm tra phần thưởng.

Như vậy, behavioral probe không thay thế hard oracle mà bổ sung một lớp đánh giá trải nghiệm. Hard oracle xác định dungeon có hợp lệ về mặt cơ học hay không; P-CBS cho biết dungeon đó có “đọc được” và “chơi được” dưới những kiểu hành vi người chơi khác nhau hay không.

## 6.1 Graph-to-Grid Cross-Attention — vị trí trên ảnh và cơ chế hoạt động (chi tiết kỹ thuật)

Vị trí trên sơ đồ:

- Module `Graph-to-Grid Cross-Attention` xuất hiện trong Block IV (Latent diffusion) — cụ thể trong phần decoder/attention stack nơi `CrossAttention` được gán nhãn thêm `SpatialGraphConditioner`/`GraphToGrid`. Trên ảnh nó nằm trong khối attention của decoder (phần phải trung tâm của Block IV), và được minh họa dưới dạng một cross-attention riêng biệt nối grid queries với graph tokens (xem biểu tượng attention trong decoder level, gần `SpatialGraphConditioner`).

Tham chiếu cài đặt:

- Code chính: `src/core/graph_grid_attention.py` — chứa `GraphToGridCrossAttention`, `SpatialGraphConditioner`, `RoomTopologyConditioner`.
- Điểm tích hợp: `src/core/latent_diffusion.py` tại các `AttentionBlock`/`CrossAttention` nơi attention có thể nhận K/V từ `fused condition c` hoặc trực tiếp từ graph node tokens sau bước message-passing.

Cơ chế hoạt động (chi tiết):

1. Inputs

   - Grid features: feature map từ decoder/encoder tại resolution r, kích thước `[B, C, H, W]`. Trước attention, map được flatten thành sequence queries `Q` có shape `[B, L_q, d]` (L_q = H*W).
   - Graph tokens: node embeddings `N` sau GNN / GraphGPS, shape `[B, N_nodes, d]`. Trước attention, node tokens có thể được enrich bằng `tpe` (topological positional encodings) và node-level scalar features (degree, locked/goal flags).
   - (Tuỳ option) room_topology_map raster: được đưa vào SpatialGraphConditioner như thêm bias hoặc gating vào Q hoặc vào attention logits.
2. Preprocessing / enrichment

   - Node tokens chạy qua một nhỏ MLP hoặc linear layer để chuẩn hóa kích thước d nếu cần.
   - `graph_pe` (positional encodings) có thể được cộng vào node embeddings hoặc dùng để tạo attention bias.
   - Nếu node count lớn, module có thể sử dụng một chế độ giảm tỉ lệ (`linear_hedgehog`) — cơ chế tuyến tính hóa tương tác để giữ chi phí tính toán trong ngưỡng.
3. Attention computation

   - Q, K, V được dựa trên: Q = Linear_Q(grid_features), K = Linear_K(node_tokens), V = Linear_V(node_tokens).
   - logits = Q K^T / sqrt(d) + topology_bias
   - weights = softmax(logits) hoặc hedgehog-softmax variant (nếu dùng hedgehog kernel)
   - output = weights V
4. Integration back to grid

   - output được reshape trở lại `[B, C, H, W]` (hoặc fused với original feature bằng residual + FFN).
   - Có thể thêm gate học được α: `feature' = α * attention_output + (1-α) * feature_before` để cho phép model học mức độ ảnh hưởng của graph lên từng vị trí.
5. Vai trò / tác dụng

   - Cho phép từng ô grid hỏi trực tiếp thông tin node-level (ví dụ: node là door, locked, goal) để đồng bộ hóa semantic tile decisions với mission-level constraints.
   - Bổ sung thông tin không gian-phi cục bộ mà `room_topology_map` raster có thể không mã hóa rõ ràng (ví dụ edge types, remote key locations).
   - Hữu dụng để đảm bảo các tile mang trách nhiệm nhiệm vụ (như đặt key/lock) được thống nhất với graph-level plan.
6. Hiệu năng và fallback

   - Khi số node quá lớn hoặc graph có đặc điểm scale-khó (n_nodes ≫ L_q), module có chế độ tuyến tính (approximate attention / hedgehog) để giảm O(N*L_q) xuống chi phí thấp hơn.
   - Nếu graph tokens thiếu hoặc bị cấu hình tắt, hệ thống vẫn hoạt động dựa trên `c_local` và `room_topology_map` (degrades gracefully).
7. Gợi ý đọc code

   - Xem `GraphToGridCrossAttention` trong `src/core/graph_grid_attention.py` để thấy chi tiết việc chuẩn hóa token, tạo bias từ TPE và lựa chọn kernel (softmax vs hedgehog). Trong `src/core/latent_diffusion.py`, tìm nơi `CrossAttention` được gọi trong decoder để thấy điểm tích hợp.

---

Nếu bạn muốn, tôi sẽ tiếp tục bằng cách: a) chèn trích đoạn code cụ thể (hàm/class) từ `src/core/graph_grid_attention.py` vào MD để minh họa chi tiết tham số và API, và b) đặt các link file trong phần tham chiếu để độc giả click trực tiếp tới hàm. Bạn muốn tôi thêm code snippet hay chỉ giữ mô tả khái quát này?
