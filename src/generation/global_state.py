"""
Feature 6: Global State System
===============================
Multi-room gimmicks: water level changes, switches affecting distant rooms.

Problem:
    Current system treats rooms independently. Cannot express:
    - Water level that affects multiple rooms
    - Switch in room A that opens door in room B
    - Bomb that creates passage between distant rooms

Solution:
    - Global State Tags: Propagate state through mission graph
    - Conditional Rendering: Re-render rooms when global state changes
    - State Dependencies: Track which rooms depend on which state
    - Multi-pass Generation: Generate → propagate state → re-generate affected rooms

Integration Point: In generate_dungeon, add state propagation between rooms
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set, Any, FrozenSet
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

class GlobalStateType(Enum):
    """Types of global state that affect multiple rooms."""
    WATER_LEVEL = "water_level"  # High/low water affects traversal
    SWITCH_LEVER = "switch_lever"  # Lever in one room affects doors elsewhere
    BOMB_WALL = "bomb_wall"  # Bombable wall creates new connection
    KEY_GATE = "key_gate"  # Boss key opens multiple doors
    LIGHT_LEVEL = "light_level"  # Torch lights multiple rooms


@dataclass
class GlobalStateVariable:
    """A single global state variable."""
    name: str
    state_type: GlobalStateType
    initial_value: Any
    current_value: Any
    affected_rooms: Set[int] = field(default_factory=set)  # Room IDs affected by this state
    conditions: Dict[str, Any] = field(default_factory=dict)  # When state changes


@dataclass
class StateTransition:
    """A state change triggered by room events."""
    from_room: int  # Room where trigger occurs
    trigger_condition: str  # "switch_pulled", "boss_defeated", etc.
    state_changes: Dict[str, Any]  # {state_variable_name: new_value}
    affected_rooms: Set[int]  # Rooms that need re-generation


@dataclass
class RoomStateDependency:
    """Tracks how a room depends on global state."""
    room_id: int
    required_states: Dict[str, Any]  # {state_var: required_value}
    optional_states: Dict[str, List[Any]] = field(default_factory=dict)  # Alternative values


class GlobalStateManager:
    """
    Manages global state propagation across multi-room dungeon.
    
    Core Concepts:
    1. State Variables: Global flags/values (water_high, switch_1_on)
    2. State Transitions: Events that change state (pull_switch → switch_1_on = True)
    3. Room Dependencies: Which rooms care about which state
    4. Conditional Rendering: Re-render rooms when relevant state changes
    
    Example Use Case:
        # Define state: Water level starts high
        manager.add_state("water_level", GlobalStateType.WATER_LEVEL, initial_value="high")
        
        # Room 3 has switch that lowers water
        manager.add_transition(
            from_room=3,
            trigger="switch_pulled",
            state_changes={"water_level": "low"},
            affected_rooms={5, 7, 9}  # These rooms have different layout when water is low
        )
        
        # Generate dungeon with water high initially
        dungeon = pipeline.generate_dungeon(...)
        
        # Simulate pulling switch
        manager.apply_transition(room_id=3, trigger="switch_pulled")
        
        # Re-render affected rooms with water low
        for room_id in [5, 7, 9]:
            updated_room = pipeline.regenerate_room(room_id, global_state=manager.get_state())
    """
    
    def __init__(self):
        self.state_variables: Dict[str, GlobalStateVariable] = {}
        self.transitions: List[StateTransition] = []
        self.room_dependencies: Dict[int, RoomStateDependency] = {}
        self.state_history: List[Dict[str, Any]] = []  # For undo/replay
    
    def add_state_variable(
        self,
        name: str,
        state_type: GlobalStateType,
        initial_value: Any
    ):
        """Register a global state variable."""
        self.state_variables[name] = GlobalStateVariable(
            name=name,
            state_type=state_type,
            initial_value=initial_value,
            current_value=initial_value
        )
        logger.debug(f"Added global state: {name} = {initial_value}")
    
    def add_transition(
        self,
        from_room: int,
        trigger_condition: str,
        state_changes: Dict[str, Any],
        affected_rooms: Set[int]
    ):
        """Define a state transition."""
        transition = StateTransition(
            from_room=from_room,
            trigger_condition=trigger_condition,
            state_changes=state_changes,
            affected_rooms=affected_rooms
        )
        self.transitions.append(transition)
        
        # Update affected rooms in state variables
        for state_name in state_changes.keys():
            if state_name in self.state_variables:
                self.state_variables[state_name].affected_rooms.update(affected_rooms)
        
        logger.debug(
            f"Added transition: Room {from_room} -> {trigger_condition} "
            f"changes {state_changes}, affects rooms {affected_rooms}"
        )
    
    def set_room_dependency(
        self,
        room_id: int,
        required_states: Dict[str, Any],
        optional_states: Optional[Dict[str, List[Any]]] = None
    ):
        """Define how a room depends on global state."""
        self.room_dependencies[room_id] = RoomStateDependency(
            room_id=room_id,
            required_states=required_states,
            optional_states=optional_states or {}
        )
    
    def apply_transition(
        self,
        room_id: int,
        trigger: str
    ) -> Set[int]:
        """
        Apply a state transition and return affected rooms.
        
        Args:
            room_id: Room where trigger occurs
            trigger: Trigger condition name
        
        Returns:
            Set of room IDs that need re-generation
        """
        # Find matching transition
        matching_transitions = [
            t for t in self.transitions
            if t.from_room == room_id and t.trigger_condition == trigger
        ]
        
        if not matching_transitions:
            logger.warning(f"No transition found for room {room_id}, trigger {trigger}")
            return set()
        
        # Save current state to history
        current_state = {name: var.current_value for name, var in self.state_variables.items()}
        self.state_history.append(current_state)
        
        # Apply all matching transitions
        affected_rooms = set()
        for transition in matching_transitions:
            # Update state variables
            for state_name, new_value in transition.state_changes.items():
                if state_name in self.state_variables:
                    old_value = self.state_variables[state_name].current_value
                    self.state_variables[state_name].current_value = new_value
                    logger.info(f"State change: {state_name} {old_value} → {new_value}")
                else:
                    logger.warning(f"Unknown state variable: {state_name}")
            
            affected_rooms.update(transition.affected_rooms)
        
        logger.info(f"Transition applied: {len(affected_rooms)} rooms affected")
        return affected_rooms
    
    def get_state(self) -> Dict[str, Any]:
        """Get current global state snapshot."""
        return {name: var.current_value for name, var in self.state_variables.items()}
    
    def get_room_state(self, room_id: int) -> Dict[str, Any]:
        """Get state relevant to a specific room."""
        if room_id not in self.room_dependencies:
            return {}
        
        dependency = self.room_dependencies[room_id]
        room_state = {}
        
        # Get required states
        for state_name in dependency.required_states.keys():
            if state_name in self.state_variables:
                room_state[state_name] = self.state_variables[state_name].current_value
        
        # Get optional states
        for state_name in dependency.optional_states.keys():
            if state_name in self.state_variables:
                room_state[state_name] = self.state_variables[state_name].current_value
        
        return room_state
    
    def reset_state(self):
        """Reset all state to initial values."""
        for var in self.state_variables.values():
            var.current_value = var.initial_value
        self.state_history.clear()
        logger.info("Global state reset to initial values")
    
    def undo_last_transition(self) -> bool:
        """Undo last state change."""
        if not self.state_history:
            return False
        
        # Restore previous state
        previous_state = self.state_history.pop()
        for state_name, value in previous_state.items():
            if state_name in self.state_variables:
                self.state_variables[state_name].current_value = value
        
        logger.info("Undid last state transition")
        return True


# ============================================================================
# STATE-AWARE ROOM GENERATOR
# ============================================================================

class StateAwareRoomGenerator:
    """
    Extends room generation to respect global state.
    
    Key Innovation:
    - Inject global state as additional conditioning
    - Modify tile generation based on state (e.g., water level)
    - Support conditional re-rendering
    """
    
    def __init__(self, base_pipeline):
        """
        Args:
            base_pipeline: NeuralSymbolicDungeonPipeline instance
        """
        self.pipeline = base_pipeline
        self.state_manager = GlobalStateManager()
    
    def generate_room_with_state(
        self,
        room_id: int,
        neighbor_latents: Dict,
        graph_context: Dict,
        global_state: Dict[str, Any],
        **kwargs
    ):
        """
        Generate room conditioned on global state.
        
        Args:
            room_id: Room identifier
            neighbor_latents: Neighboring room latents
            graph_context: Graph context data
            global_state: Current global state dict
            **kwargs: Additional args for generate_room
        
        Returns:
            RoomGenerationResult
        """
        # Encode global state as additional conditioning
        state_embedding = self._encode_global_state(global_state)
        
        # Inject into graph context
        graph_context = graph_context.copy()
        if 'global_state' not in graph_context:
            graph_context['global_state'] = state_embedding
        
        # Generate room with state conditioning
        result = self.pipeline.generate_room(
            neighbor_latents=neighbor_latents,
            graph_context=graph_context,
            room_id=room_id,
            **kwargs
        )
        
        # Post-process based on state
        result.room_grid = self._apply_state_modifications(
            result.room_grid,
            global_state
        )
        
        return result
    
    def regenerate_with_state_change(
        self,
        room_id: int,
        old_state: Dict[str, Any],
        new_state: Dict[str, Any],
        cached_rooms: Dict[int, Any],
        graph_context: Dict
    ):
        """
        Re-generate room after state change.
        
        Uses cached neighbor latents when possible to maintain consistency.
        """
        # Get neighbor latents from cache
        neighbor_latents = {}
        # ... extract from cached_rooms ...
        
        # Generate with new state
        result = self.generate_room_with_state(
            room_id=room_id,
            neighbor_latents=neighbor_latents,
            graph_context=graph_context,
            global_state=new_state
        )
        
        return result
    
    def _encode_global_state(self, global_state: Dict[str, Any]) -> np.ndarray:
        """Encode global state as feature vector."""
        # Simple encoding: hash state to fixed-size vector
        # In production, use learned embedding
        
        embedding = np.zeros(16, dtype=np.float32)
        
        # Water level: 0 = low, 1 = high
        if 'water_level' in global_state:
            embedding[0] = 1.0 if global_state['water_level'] == 'high' else 0.0
        
        # Switches: binary flags
        for i, key in enumerate(sorted([k for k in global_state.keys() if k.startswith('switch_')])):
            if i < 8:
                embedding[i + 1] = 1.0 if global_state[key] else 0.0
        
        # Light level: continuous [0, 1]
        if 'light_level' in global_state:
            embedding[9] = float(global_state['light_level'])
        
        return embedding
    
    def _apply_state_modifications(
        self,
        room_grid: np.ndarray,
        global_state: Dict[str, Any]
    ) -> np.ndarray:
        """
        Modify room grid based on global state.
        
        Examples:
        - Water high → fill low areas with water
        - Light low → convert visual palette to dark
        """
        modified = room_grid.copy()
        
        # Water level modification
        if 'water_level' in global_state:
            water_level = global_state['water_level']
            if water_level == 'high':
                # Fill floor tiles in bottom half with water (tile ID 40)
                H = modified.shape[0]
                water_line = H // 2
                floor_mask = (modified[water_line:] == 1)  # FLOOR_ID
                modified[water_line:][floor_mask] = 40  # WATER_ID
        
        # Additional state modifications...
        
        return modified


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================

"""
# In src/pipeline/dungeon_pipeline.py:

from src.generation.global_state import (
    GlobalStateManager,
    GlobalStateType,
    StateAwareRoomGenerator
)

class NeuralSymbolicDungeonPipeline:
    def __init__(self, ...):
        # ... existing init ...
        self.global_state_manager = GlobalStateManager()
        self.state_aware_generator = StateAwareRoomGenerator(self)
    
    def generate_dungeon_with_global_state(
        self,
        mission_graph: nx.Graph,
        state_config: Optional[Dict] = None,
        **kwargs
    ):
        '''
        Generate dungeon with multi-room gimmicks.
        
        Args:
            mission_graph: Mission graph
            state_config: {
                'variables': [
                    {'name': 'water_level', 'type': 'water_level', 'initial': 'high'},
                    {'name': 'switch_1', 'type': 'switch_lever', 'initial': False}
                ],
                'transitions': [
                    {
                        'from_room': 3,
                        'trigger': 'switch_pulled',
                        'changes': {'water_level': 'low'},
                        'affects': [5, 7, 9]
                    }
                ]
            }
        '''
        # Setup global state
        if state_config:
            for var_config in state_config.get('variables', []):
                self.global_state_manager.add_state_variable(
                    name=var_config['name'],
                    state_type=GlobalStateType(var_config['type']),
                    initial_value=var_config['initial']
                )
            
            for trans_config in state_config.get('transitions', []):
                self.global_state_manager.add_transition(
                    from_room=trans_config['from_room'],
                    trigger_condition=trans_config['trigger'],
                    state_changes=trans_config['changes'],
                    affected_rooms=set(trans_config['affects'])
                )
        
        # Generate rooms with initial state
        rooms = {}
        for room_id in nx.topological_sort(mission_graph):
            global_state = self.global_state_manager.get_state()
            
            room_result = self.state_aware_generator.generate_room_with_state(
                room_id=room_id,
                neighbor_latents=self._get_neighbor_latents(room_id, mission_graph, rooms),
                graph_context=self._prepare_graph_context(mission_graph),
                global_state=global_state,
                **kwargs
            )
            
            rooms[room_id] = room_result
        
        return rooms, self.global_state_manager


# Usage example - dungeon with water level puzzle:

state_config = {
    'variables': [
        {'name': 'water_level', 'type': 'water_level', 'initial': 'high'}
    ],
    'transitions': [
        {
            'from_room': 3,  # Room 3 has water control switch
            'trigger': 'switch_pulled',
            'changes': {'water_level': 'low'},
            'affects': [4, 5, 6]  # These rooms have passages revealed when water lowers
        }
    ]
}

rooms, state_manager = pipeline.generate_dungeon_with_global_state(
    mission_graph=graph,
    state_config=state_config
)

# Simulate player pulling switch in room 3
affected_rooms = state_manager.apply_transition(room_id=3, trigger='switch_pulled')

# Re-generate affected rooms with new state
for room_id in affected_rooms:
    new_state = state_manager.get_state()
    rooms[room_id] = pipeline.state_aware_generator.generate_room_with_state(
        room_id=room_id,
        neighbor_latents=...,
        graph_context=...,
        global_state=new_state
    )
"""
