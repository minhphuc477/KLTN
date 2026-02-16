"""
Entity Spawner
Converts abstract mission graph attributes to concrete spatial gameplay entities.

This addresses the thesis defense concern: "Your system generates pretty layouts but no gameplay - 
where are the enemies, keys, chests?"
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of spawnable gameplay entities."""
    ENEMY_WEAK = "enemy_weak"
    ENEMY_STRONG = "enemy_strong"
    ENEMY_BOSS = "enemy_boss"
    KEY = "key"
    CHEST = "chest"
    HEALTH_POTION = "health_potion"
    MANA_POTION = "mana_potion"
    TRAP = "trap"
    NPC = "npc"
    ITEM_SWORD = "item_sword"
    ITEM_SHIELD = "item_shield"


@dataclass
class Entity:
    """Represents a gameplay entity in the dungeon."""
    entity_type: EntityType
    x: int
    y: int
    room_id: int
    properties: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary for export."""
        return {
            'type': self.entity_type.value,
            'position': {'x': self.x, 'y': self.y},
            'room_id': self.room_id,
            'properties': self.properties
        }


@dataclass
class RoomSemantics:
    """Semantic information about a room from mission graph."""
    node_id: int
    room_type: str  # 'start', 'combat', 'treasure', 'boss', 'puzzle', 'safe'
    difficulty: float  # 0.0 to 1.0
    has_key: bool = False
    has_treasure: bool = False
    is_critical_path: bool = False
    tension_value: float = 0.5


class EntitySpawner:
    """
    Spawns gameplay entities in generated rooms based on semantic attributes.
    
    Design Principles:
    - Enemies spawn away from doors (prevent spawn-camping)
    - Keys spawn in visible but protected locations
    - Density scales with room difficulty
    - Spatial clustering avoided (even distribution)
    - Boss rooms get single powerful enemy
    - Treasure rooms have chests + weak guards
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: Spawning parameters
                {
                    'enemy_density': 0.15,  # enemies per floor tile
                    'min_enemy_distance': 3,  # tiles from doors
                    'min_entity_spacing': 2,  # tiles between entities
                    'floor_tile_id': 0,
                    'wall_tile_id': 1,
                    'door_tile_ids': [2, 3, 4]
                }
        """
        self.config = config or {}
        self.enemy_density = self.config.get('enemy_density', 0.15)
        self.min_enemy_distance = self.config.get('min_enemy_distance', 3)
        self.min_entity_spacing = self.config.get('min_entity_spacing', 2)
        
        self.FLOOR_ID = self.config.get('floor_tile_id', 0)
        self.WALL_ID = self.config.get('wall_tile_id', 1)
        self.DOOR_IDS = self.config.get('door_tile_ids', [2, 3, 4])
    
    def spawn_entities(
        self,
        room_grid: np.ndarray,
        room_semantics: RoomSemantics,
        room_bounds: Tuple[int, int, int, int],
        seed: Optional[int] = None
    ) -> List[Entity]:
        """
        Generate entity list for a single room.
        
        Args:
            room_grid: (H, W) tile array for the room
            room_semantics: Semantic information from mission graph
            room_bounds: (x_min, y_min, x_max, y_max) in global coordinates
            seed: Random seed for deterministic spawning
        
        Returns:
            List of Entity instances with world positions
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        entities = []
        
        # Step 1: Identify valid spawn points
        spawn_candidates = self._find_spawn_candidates(room_grid)
        
        if len(spawn_candidates) == 0:
            logger.warning(f"Room {room_semantics.node_id} has no valid spawn points")
            return []
        
        logger.debug(f"Room {room_semantics.node_id} ({room_semantics.room_type}): {len(spawn_candidates)} spawn candidates")
        
        # Step 2: Spawn based on room type
        if room_semantics.room_type == 'start':
            # Start room: no enemies, maybe health potion
            entities.extend(self._spawn_start_room(
                spawn_candidates, room_semantics, room_bounds
            ))
        
        elif room_semantics.room_type == 'combat':
            # Combat room: spawn enemies based on difficulty
            entities.extend(self._spawn_combat_room(
                spawn_candidates, room_semantics, room_bounds
            ))
        
        elif room_semantics.room_type == 'boss':
            # Boss room: single powerful enemy in center
            entities.extend(self._spawn_boss_room(
                spawn_candidates, room_semantics, room_bounds
            ))
        
        elif room_semantics.room_type == 'treasure':
            # Treasure room: chest + maybe weak guard
            entities.extend(self._spawn_treasure_room(
                spawn_candidates, room_semantics, room_bounds
            ))
        
        elif room_semantics.room_type == 'puzzle':
            # Puzzle room: maybe NPC or key
            entities.extend(self._spawn_puzzle_room(
                spawn_candidates, room_semantics, room_bounds
            ))
        
        elif room_semantics.room_type == 'safe':
            # Safe room: healing items, no enemies
            entities.extend(self._spawn_safe_room(
                spawn_candidates, room_semantics, room_bounds
            ))
        
        # Step 3: Spawn key if room has one
        if room_semantics.has_key and len(spawn_candidates) > 0:
            key_entity = self._spawn_key(
                spawn_candidates, entities, room_semantics, room_bounds
            )
            if key_entity:
                entities.append(key_entity)
        
        logger.info(f"Room {room_semantics.node_id}: Spawned {len(entities)} entities")
        
        return entities
    
    def _find_spawn_candidates(self, room_grid: np.ndarray) -> List[Tuple[int, int]]:
        """Identify valid spawn positions (floor tiles away from doors)."""
        candidates = []
        H, W = room_grid.shape
        
        # Find all door positions
        door_positions = []
        for y in range(H):
            for x in range(W):
                if room_grid[y, x] in self.DOOR_IDS:
                    door_positions.append(np.array([y, x]))
        
        door_positions = np.array(door_positions) if door_positions else np.array([]).reshape(0, 2)
        
        for y in range(H):
            for x in range(W):
                # Must be floor
                if room_grid[y, x] != self.FLOOR_ID:
                    continue
                
                # Must be away from doors
                if len(door_positions) > 0:
                    distances = np.linalg.norm(door_positions - [y, x], axis=1)
                    if np.min(distances) < self.min_enemy_distance:
                        continue
                
                # Must not be on edge (avoid walls)
                if x <= 1 or x >= W-2 or y <= 1 or y >= H-2:
                    continue
                
                candidates.append((x, y))
        
        return candidates
    
    def _spawn_start_room(
        self, candidates: List[Tuple[int, int]], 
        semantics: RoomSemantics,
        bounds: Tuple[int, int, int, int]
    ) -> List[Entity]:
        """Spawn entities for start room (safe, maybe health potion)."""
        entities = []
        
        # 30% chance for starting health potion
        if random.random() < 0.3 and candidates:
            pos = random.choice(candidates)
            entities.append(Entity(
                EntityType.HEALTH_POTION,
                pos[0] + bounds[0],
                pos[1] + bounds[1],
                semantics.node_id,
                {'heal_amount': 50}
            ))
        
        return entities
    
    def _spawn_combat_room(
        self, candidates: List[Tuple[int, int]],
        semantics: RoomSemantics,
        bounds: Tuple[int, int, int, int]
    ) -> List[Entity]:
        """Spawn enemies for combat room."""
        entities = []
        
        num_enemies = self._calculate_enemy_count(len(candidates), semantics.difficulty)
        
        enemy_positions = self._select_spaced_positions(
            candidates, num_enemies
        )
        
        for pos in enemy_positions:
            enemy_type = self._select_enemy_type(semantics.difficulty)
            hp = self._get_enemy_hp(enemy_type, semantics.difficulty)
            
            entities.append(Entity(
                enemy_type,
                pos[0] + bounds[0],
                pos[1] + bounds[1],
                semantics.node_id,
                {
                    'hp': hp,
                    'damage': int(hp * 0.2),
                    'difficulty': semantics.difficulty
                }
            ))
        
        # Maybe add a health potion (10% chance)
        if random.random() < 0.1 and len(enemy_positions) < len(candidates):
            remaining = [c for c in candidates if c not in enemy_positions]
            if remaining:
                pos = random.choice(remaining)
                entities.append(Entity(
                    EntityType.HEALTH_POTION,
                    pos[0] + bounds[0],
                    pos[1] + bounds[1],
                    semantics.node_id,
                    {'heal_amount': 30}
                ))
        
        return entities
    
    def _spawn_boss_room(
        self, candidates: List[Tuple[int, int]],
        semantics: RoomSemantics,
        bounds: Tuple[int, int, int, int]
    ) -> List[Entity]:
        """Spawn boss enemy in room center."""
        entities = []
        
        # Find center position
        center_pos = self._select_central_position(candidates)
        
        entities.append(Entity(
            EntityType.ENEMY_BOSS,
            center_pos[0] + bounds[0],
            center_pos[1] + bounds[1],
            semantics.node_id,
            {
                'hp': 200,
                'damage': 30,
                'boss_id': semantics.node_id,
                'name': f'Boss_{semantics.node_id}'
            }
        ))
        
        return entities
    
    def _spawn_treasure_room(
        self, candidates: List[Tuple[int, int]],
        semantics: RoomSemantics,
        bounds: Tuple[int, int, int, int]
    ) -> List[Entity]:
        """Spawn chest and optional guard."""
        entities = []
        
        # Chest (guaranteed)
        chest_pos = random.choice(candidates)
        entities.append(Entity(
            EntityType.CHEST,
            chest_pos[0] + bounds[0],
            chest_pos[1] + bounds[1],
            semantics.node_id,
            {
                'loot': 'gold' if random.random() < 0.7 else 'rare_item',
                'amount': random.randint(50, 150)
            }
        ))
        
        # Weak guard (50% chance)
        if random.random() < 0.5:
            remaining = [c for c in candidates if c != chest_pos]
            if remaining:
                guard_pos = random.choice(remaining)
                entities.append(Entity(
                    EntityType.ENEMY_WEAK,
                    guard_pos[0] + bounds[0],
                    guard_pos[1] + bounds[1],
                    semantics.node_id,
                    {'hp': 30, 'damage': 8}
                ))
        
        return entities
    
    def _spawn_puzzle_room(
        self, candidates: List[Tuple[int, int]],
        semantics: RoomSemantics,
        bounds: Tuple[int, int, int, int]
    ) -> List[Entity]:
        """Spawn puzzle-related entities (NPC, item)."""
        entities = []
        
        # 40% chance for NPC
        if random.random() < 0.4 and candidates:
            pos = random.choice(candidates)
            entities.append(Entity(
                EntityType.NPC,
                pos[0] + bounds[0],
                pos[1] + bounds[1],
                semantics.node_id,
                {
                    'name': f'NPC_{semantics.node_id}',
                    'dialogue': 'Solve the puzzle to proceed.'
                }
            ))
        
        return entities
    
    def _spawn_safe_room(
        self, candidates: List[Tuple[int, int]],
        semantics: RoomSemantics,
        bounds: Tuple[int, int, int, int]
    ) -> List[Entity]:
        """Spawn healing items in safe room."""
        entities = []
        
        # Health potion (70% chance)
        if random.random() < 0.7 and candidates:
            pos = random.choice(candidates)
            entities.append(Entity(
                EntityType.HEALTH_POTION,
                pos[0] + bounds[0],
                pos[1] + bounds[1],
                semantics.node_id,
                {'heal_amount': 75}
            ))
        
        # Mana potion (50% chance)
        if random.random() < 0.5 and len(candidates) > 1:
            remaining = [c for c in candidates]
            pos = random.choice(remaining)
            entities.append(Entity(
                EntityType.MANA_POTION,
                pos[0] + bounds[0],
                pos[1] + bounds[1],
                semantics.node_id,
                {'mana_amount': 50}
            ))
        
        return entities
    
    def _spawn_key(
        self, candidates: List[Tuple[int, int]],
        existing_entities: List[Entity],
        semantics: RoomSemantics,
        bounds: Tuple[int, int, int, int]
    ) -> Optional[Entity]:
        """Spawn key in visible protected location."""
        # Exclude positions already occupied
        occupied = [(e.x - bounds[0], e.y - bounds[1]) for e in existing_entities]
        available = [c for c in candidates if c not in occupied]
        
        if not available:
            return None
        
        key_pos = random.choice(available)
        return Entity(
            EntityType.KEY,
            key_pos[0] + bounds[0],
            key_pos[1] + bounds[1],
            semantics.node_id,
            {
                'key_id': semantics.node_id,
                'unlocks': 'door' if random.random() < 0.8 else 'chest'
            }
        )
    
    def _calculate_enemy_count(self, num_candidates: int, difficulty: float) -> int:
        """Calculate appropriate number of enemies for room.
        
        ANTI-SPAM CONSTRAINT: Cap enemy count to prevent genetic algorithm
        from converging to trivial "enemy spam" solutions. Forces balanced
        difficulty through navigation complexity, not raw enemy density.
        
        Constraints:
        - Max 25% of floor space occupied
        - Absolute cap of 8 enemies per room (Zelda standard)
        - Minimum spacing requirement enforced by Poisson disk sampling
        """
        base_count = int(num_candidates * self.enemy_density)
        difficulty_mult = 0.5 + difficulty  # 0.5 to 1.5 multiplier
        final_count = int(base_count * difficulty_mult)
        
        # CRITICAL: Multi-objective constraint to prevent enemy spam
        spatial_cap = num_candidates // 4  # Max 25% of space
        absolute_cap = 8  # Zelda-style enemy count limit
        
        return max(1, min(final_count, spatial_cap, absolute_cap))
    
    def _select_enemy_type(self, difficulty: float) -> EntityType:
        """Choose enemy type based on difficulty."""
        if difficulty < 0.4:
            return EntityType.ENEMY_WEAK
        elif difficulty < 0.8:
            return EntityType.ENEMY_STRONG if random.random() < 0.6 else EntityType.ENEMY_WEAK
        else:
            # High difficulty: mostly strong enemies
            choices = [EntityType.ENEMY_STRONG, EntityType.ENEMY_WEAK]
            weights = [0.7, 0.3]
            return random.choices(choices, weights=weights)[0]
    
    def _get_enemy_hp(self, enemy_type: EntityType, difficulty: float) -> int:
        """Calculate enemy HP based on type and difficulty."""
        base_hp = {
            EntityType.ENEMY_WEAK: 20,
            EntityType.ENEMY_STRONG: 50,
            EntityType.ENEMY_BOSS: 200
        }.get(enemy_type, 20)
        
        return int(base_hp * (0.8 + 0.4 * difficulty))  # ±20% variation
    
    def _select_spaced_positions(
        self, candidates: List[Tuple[int, int]],
        num_positions: int
    ) -> List[Tuple[int, int]]:
        """Select positions with even spacing (Poisson disk sampling)."""
        if num_positions == 0 or not candidates:
            return []
        
        selected = []
        available = candidates.copy()
        random.shuffle(available)
        
        for pos in available:
            if len(selected) >= num_positions:
                break
            
            # Check spacing constraint
            too_close = False
            for existing_pos in selected:
                dist = np.linalg.norm(np.array(pos) - np.array(existing_pos))
                if dist < self.min_entity_spacing:
                    too_close = True
                    break
            
            if not too_close:
                selected.append(pos)
        
        # If we didn't get enough, relax constraint
        if len(selected) < num_positions:
            remaining = [c for c in available if c not in selected]
            needed = num_positions - len(selected)
            selected.extend(remaining[:needed])
        
        return selected
    
    def _select_central_position(
        self, candidates: List[Tuple[int, int]]
    ) -> Tuple[int, int]:
        """Select position closest to room center (for boss)."""
        if not candidates:
            return (0, 0)
        
        center = np.mean(candidates, axis=0)
        distances = [np.linalg.norm(np.array(c) - center) for c in candidates]
        return candidates[np.argmin(distances)]


def create_room_semantics_from_graph(
    mission_graph: Dict,
    node_id: int,
    tension_curve: Optional[List[float]] = None
) -> RoomSemantics:
    """
    Extract room semantics from mission graph node.
    
    Args:
        mission_graph: Complete mission graph
        node_id: Node to extract semantics for
        tension_curve: Optional tension values for rooms
    
    Returns:
        RoomSemantics instance
    """
    node_data = mission_graph['nodes'].get(node_id, {})
    
    # Infer room type from node attributes
    room_type = node_data.get('type', 'combat')
    if node_id == 0:
        room_type = 'start'
    elif node_data.get('is_boss', False):
        room_type = 'boss'
    elif node_data.get('has_treasure', False):
        room_type = 'treasure'
    
    # Get difficulty
    difficulty = node_data.get('difficulty', 0.5)
    if tension_curve and node_id < len(tension_curve):
        difficulty = tension_curve[node_id]
    
    return RoomSemantics(
        node_id=node_id,
        room_type=room_type,
        difficulty=difficulty,
        has_key=node_data.get('has_key', False),
        has_treasure=node_data.get('has_treasure', False),
        is_critical_path=node_data.get('is_critical_path', False),
        tension_value=difficulty
    )


def spawn_all_entities(
    dungeon_grid: np.ndarray,
    mission_graph: Dict,
    layout_map: Dict[int, Tuple[int, int, int, int]],
    config: Optional[Dict] = None,
    seed: Optional[int] = None
) -> List[Entity]:
    """
    Spawn entities for entire dungeon.
    
    Args:
        dungeon_grid: Full (H, W) dungeon grid
        mission_graph: Complete mission graph
        layout_map: Node ID -> (x_min, y_min, x_max, y_max) mapping
        config: Spawner configuration
        seed: Random seed
    
    Returns:
        List of all entities with global coordinates
    """
    spawner = EntitySpawner(config)
    all_entities = []
    
    logger.info(f"Spawning entities for {len(layout_map)} rooms...")
    
    for node_id, room_bounds in layout_map.items():
        x_min, y_min, x_max, y_max = room_bounds
        
        # Extract room subgrid
        room_grid = dungeon_grid[y_min:y_max+1, x_min:x_max+1]
        
        # Get room semantics
        semantics = create_room_semantics_from_graph(mission_graph, node_id)
        
        # Spawn entities
        room_seed = (seed + node_id) if seed is not None else None
        entities = spawner.spawn_entities(
            room_grid,
            semantics,
            room_bounds,
            room_seed
        )
        
        all_entities.extend(entities)
    
    logger.info(f"Total entities spawned: {len(all_entities)}")
    
    return all_entities


def export_entities_to_json(entities: List[Entity], filepath: str) -> None:
    """Export entity list to JSON file for game engine."""
    data = {
        'entities': [e.to_dict() for e in entities],
        'count': len(entities),
        'types': {
            entity_type.value: sum(1 for e in entities if e.entity_type == entity_type)
            for entity_type in EntityType
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Exported {len(entities)} entities to {filepath}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy dungeon grid
    dungeon_grid = np.zeros((64, 64), dtype=int)
    dungeon_grid[0::16, :] = 1  # Walls
    dungeon_grid[:, 0::16] = 1
    
    # Create dummy mission graph
    mission_graph = {
        'nodes': {
            0: {'type': 'start'},
            1: {'type': 'combat', 'difficulty': 0.5},
            2: {'type': 'boss', 'is_boss': True}
        },
        'edges': [(0, 1), (1, 2)]
    }
    
    # Create layout map
    layout_map = {
        0: (0, 0, 15, 15),
        1: (16, 0, 31, 15),
        2: (32, 0, 47, 15)
    }
    
    # Spawn entities
    entities = spawn_all_entities(dungeon_grid, mission_graph, layout_map, seed=42)
    
    print(f"\nSpawned {len(entities)} entities:")
    for entity in entities:
        print(f"  {entity.entity_type.value} at ({entity.x}, {entity.y}) in room {entity.room_id}")
    
    # Export
    export_entities_to_json(entities, "test_entities.json")
