"""
KLTN Visualization - Visual Effects
====================================

Visual feedback effects for polished game feel.

Includes:
- PopEffect: Scale-up-then-down for item collection
- FlashEffect: Brief color flash for door opening
- TrailEffect: Glowing footstep trail
- RippleEffect: Expanding circle for teleportation
- PulseEffect: Rhythmic glow for objectives

All effects use delta-time for frame-rate independent animation.

"""

from __future__ import annotations

import math
from typing import Tuple, Optional, List, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

try:
    import pygame
    from pygame import Surface
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


# ==========================================
# BASE EFFECT
# ==========================================

class EffectState(Enum):
    """Effect lifecycle states."""
    STARTING = "starting"
    ACTIVE = "active"
    ENDING = "ending"
    FINISHED = "finished"


@dataclass
class BaseEffect:
    """
    Base class for visual effects.
    
    All effects have:
    - A position (grid coordinates)
    - A lifetime (how long they run)
    - An elapsed time tracker
    - Update and render methods
    """
    
    position: Tuple[int, int]  # Grid (row, col)
    lifetime: float = 1.0  # Seconds
    elapsed: float = 0.0
    
    def update(self, dt: float) -> None:
        """Update effect state."""
        self.elapsed += dt
    
    def is_active(self) -> bool:
        """Check if effect is still running."""
        return self.elapsed < self.lifetime
    
    def progress(self) -> float:
        """Get normalized progress (0.0 to 1.0)."""
        return min(1.0, self.elapsed / self.lifetime) if self.lifetime > 0 else 1.0
    
    def render(self, surface: Surface, tile_size: int, 
               camera_offset: Tuple[int, int] = (0, 0)) -> None:
        """Render the effect. Override in subclasses."""
        pass


# ==========================================
# POP EFFECT
# ==========================================

class PopEffect(BaseEffect):
    """
    Scale-up-then-down effect for item collection.
    
    Creates a satisfying "pop" visual when collecting keys, items, etc.
    The item briefly scales up then shrinks to nothing.
    """
    
    def __init__(self, position: Tuple[int, int], 
                 color: Tuple[int, int, int] = (255, 220, 100),
                 lifetime: float = 0.4):
        self.position = position
        self.color = color
        self.lifetime = lifetime
        self.elapsed = 0.0
        self.max_scale = 1.5
    
    def render(self, surface: Surface, tile_size: int,
               camera_offset: Tuple[int, int] = (0, 0)) -> None:
        if not PYGAME_AVAILABLE:
            return
        
        t = self.progress()
        
        # Scale curve: quick expand then slow shrink
        if t < 0.3:
            # Expand phase
            scale = 1.0 + (self.max_scale - 1.0) * (t / 0.3)
        else:
            # Shrink phase
            shrink_t = (t - 0.3) / 0.7
            scale = self.max_scale * (1.0 - shrink_t)
        
        scale = max(0.1, scale)
        
        # Calculate position
        row, col = self.position
        cam_x, cam_y = camera_offset
        center_x = col * tile_size + tile_size // 2 - cam_x
        center_y = row * tile_size + tile_size // 2 - cam_y
        
        # Draw expanding/shrinking circle
        radius = int(tile_size * scale / 2)
        alpha = int(255 * (1 - t))
        
        effect_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(effect_surface, (*self.color, alpha), 
                          (radius, radius), radius)
        
        surface.blit(effect_surface, 
                    (center_x - radius, center_y - radius))


# ==========================================
# FLASH EFFECT
# ==========================================

class FlashEffect(BaseEffect):
    """
    Brief color flash for door opening or damage.
    
    Creates a quick white/colored flash over a tile area.
    """
    
    def __init__(self, position: Tuple[int, int],
                 color: Tuple[int, int, int] = (255, 255, 255),
                 lifetime: float = 0.2):
        self.position = position
        self.color = color
        self.lifetime = lifetime
        self.elapsed = 0.0
    
    def render(self, surface: Surface, tile_size: int,
               camera_offset: Tuple[int, int] = (0, 0)) -> None:
        if not PYGAME_AVAILABLE:
            return
        
        t = self.progress()
        
        # Quick fade out
        alpha = int(200 * (1 - t))
        
        row, col = self.position
        cam_x, cam_y = camera_offset
        screen_x = col * tile_size - cam_x
        screen_y = row * tile_size - cam_y
        
        effect_surface = pygame.Surface((tile_size, tile_size), pygame.SRCALPHA)
        effect_surface.fill((*self.color, alpha))
        
        surface.blit(effect_surface, (screen_x, screen_y))


# ==========================================
# TRAIL EFFECT
# ==========================================

@dataclass
class TrailPoint:
    """Single point in a trail."""
    x: float
    y: float
    alpha: float
    age: float = 0.0


class TrailEffect(BaseEffect):
    """
    Glowing footstep trail for path visualization.
    
    Leaves a fading trail of glowing points along the movement path.
    """
    
    def __init__(self, color: Tuple[int, int, int] = (100, 150, 255),
                 max_points: int = 30,
                 fade_time: float = 2.0):
        self.position = (0, 0)  # Not used for trail
        self.lifetime = float('inf')  # Trails persist until cleared
        self.elapsed = 0.0
        self.color = color
        self.max_points = max_points
        self.fade_time = fade_time
        self.points: List[TrailPoint] = []
    
    def add_point(self, row: int, col: int):
        """Add a new trail point."""
        point = TrailPoint(
            x=col + 0.5,  # Center of tile
            y=row + 0.5,
            alpha=1.0
        )
        self.points.append(point)
        
        # Limit trail length
        if len(self.points) > self.max_points:
            self.points.pop(0)
    
    def update(self, dt: float) -> None:
        """Update trail points."""
        self.elapsed += dt
        
        # Age and fade points
        remaining = []
        for point in self.points:
            point.age += dt
            point.alpha = max(0, 1.0 - (point.age / self.fade_time))
            if point.alpha > 0.01:
                remaining.append(point)
        self.points = remaining
    
    def is_active(self) -> bool:
        """Trail is active as long as it has visible points."""
        return len(self.points) > 0
    
    def clear(self):
        """Clear all trail points."""
        self.points.clear()
    
    def render(self, surface: Surface, tile_size: int,
               camera_offset: Tuple[int, int] = (0, 0)) -> None:
        if not PYGAME_AVAILABLE:
            return
        
        cam_x, cam_y = camera_offset
        
        for point in self.points:
            alpha = int(point.alpha * 150)
            if alpha < 5:
                continue
            
            screen_x = int(point.x * tile_size) - cam_x
            screen_y = int(point.y * tile_size) - cam_y
            
            # Draw glowing dot
            size = int(tile_size * 0.4 * point.alpha)
            if size < 2:
                continue
            
            effect_surface = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(effect_surface, (*self.color, alpha),
                             (size, size), size)
            
            surface.blit(effect_surface, 
                        (screen_x - size, screen_y - size))


# ==========================================
# RIPPLE EFFECT
# ==========================================

class RippleEffect(BaseEffect):
    """
    Expanding circle effect for teleportation.
    
    Creates a rippling circle that expands outward and fades.
    """
    
    def __init__(self, position: Tuple[int, int],
                 color: Tuple[int, int, int] = (100, 200, 255),
                 lifetime: float = 0.6,
                 max_radius_tiles: float = 2.0):
        self.position = position
        self.color = color
        self.lifetime = lifetime
        self.elapsed = 0.0
        self.max_radius_tiles = max_radius_tiles
    
    def render(self, surface: Surface, tile_size: int,
               camera_offset: Tuple[int, int] = (0, 0)) -> None:
        if not PYGAME_AVAILABLE:
            return
        
        t = self.progress()
        
        # Expand with ease-out
        ease_t = 1 - (1 - t) ** 2  # Quadratic ease-out
        radius = int(tile_size * self.max_radius_tiles * ease_t)
        
        # Fade out
        alpha = int(180 * (1 - t))
        
        row, col = self.position
        cam_x, cam_y = camera_offset
        center_x = col * tile_size + tile_size // 2 - cam_x
        center_y = row * tile_size + tile_size // 2 - cam_y
        
        if radius > 0:
            effect_surface = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
            center = radius + 2
            
            # Draw multiple rings for depth
            for i in range(3):
                ring_radius = radius - i * 3
                ring_alpha = alpha - i * 40
                if ring_radius > 0 and ring_alpha > 0:
                    pygame.draw.circle(effect_surface, (*self.color, ring_alpha),
                                      (center, center), ring_radius, 2)
            
            surface.blit(effect_surface,
                        (center_x - center, center_y - center))


# ==========================================
# PULSE EFFECT
# ==========================================

class PulseEffect(BaseEffect):
    """
    Rhythmic glow effect for objectives (triforce, etc).
    
    Creates a pulsing glow that continuously animates.
    """
    
    def __init__(self, position: Tuple[int, int],
                 color: Tuple[int, int, int] = (255, 215, 0),
                 pulse_rate: float = 2.0):
        self.position = position
        self.color = color
        self.lifetime = float('inf')  # Pulses indefinitely
        self.elapsed = 0.0
        self.pulse_rate = pulse_rate
        self._enabled = True
    
    def set_enabled(self, enabled: bool):
        """Enable or disable the pulse."""
        self._enabled = enabled
    
    def is_active(self) -> bool:
        """Pulse is active while enabled."""
        return self._enabled
    
    def render(self, surface: Surface, tile_size: int,
               camera_offset: Tuple[int, int] = (0, 0)) -> None:
        if not PYGAME_AVAILABLE or not self._enabled:
            return
        
        # Sine wave for smooth pulsing
        pulse = (math.sin(self.elapsed * self.pulse_rate * math.pi * 2) + 1) / 2
        
        # Glow intensity varies with pulse
        alpha = int(40 + 60 * pulse)
        scale = 1.0 + 0.2 * pulse
        
        row, col = self.position
        cam_x, cam_y = camera_offset
        center_x = col * tile_size + tile_size // 2 - cam_x
        center_y = row * tile_size + tile_size // 2 - cam_y
        
        radius = int(tile_size * scale / 2)
        
        effect_surface = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
        center = radius + 2
        
        # Soft glow
        for i in range(5):
            r = radius - i * 2
            a = alpha - i * 10
            if r > 0 and a > 0:
                pygame.draw.circle(effect_surface, (*self.color, a),
                                  (center, center), r)
        
        surface.blit(effect_surface,
                    (center_x - center, center_y - center))


# ==========================================
# PARTICLE EFFECT
# ==========================================

@dataclass  
class Particle:
    """Single particle in a particle system."""
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    color: Tuple[int, int, int]
    size: float


class ParticleEffect(BaseEffect):
    """
    Particle system for sparkles, explosions, etc.
    
    Creates multiple small particles that move and fade.
    """
    
    def __init__(self, position: Tuple[int, int],
                 color: Tuple[int, int, int] = (255, 220, 100),
                 particle_count: int = 20,
                 lifetime: float = 1.0,
                 speed: float = 50.0):
        self.position = position
        self.color = color
        self.lifetime = lifetime
        self.elapsed = 0.0
        self.particles: List[Particle] = []
        
        import random
        
        # Spawn particles
        row, col = position
        center_x = col + 0.5
        center_y = row + 0.5
        
        for _ in range(particle_count):
            angle = random.uniform(0, math.pi * 2)
            velocity = random.uniform(speed * 0.5, speed)
            particle = Particle(
                x=center_x,
                y=center_y,
                vx=math.cos(angle) * velocity,
                vy=math.sin(angle) * velocity,
                life=random.uniform(lifetime * 0.5, lifetime),
                max_life=lifetime,
                color=color,
                size=random.uniform(2, 5)
            )
            self.particles.append(particle)
    
    def update(self, dt: float) -> None:
        """Update particle positions and lifetimes."""
        self.elapsed += dt
        
        gravity = 100  # Pixels per second squared
        
        remaining = []
        for p in self.particles:
            p.life -= dt
            if p.life > 0:
                p.x += p.vx * dt / 32  # Convert to tile units
                p.y += p.vy * dt / 32
                p.vy += gravity * dt / 32  # Gravity
                remaining.append(p)
        
        self.particles = remaining
    
    def is_active(self) -> bool:
        """Active while particles remain."""
        return len(self.particles) > 0
    
    def render(self, surface: Surface, tile_size: int,
               camera_offset: Tuple[int, int] = (0, 0)) -> None:
        if not PYGAME_AVAILABLE:
            return
        
        cam_x, cam_y = camera_offset
        
        for p in self.particles:
            alpha = int(255 * (p.life / p.max_life))
            if alpha < 10:
                continue
            
            screen_x = int(p.x * tile_size) - cam_x
            screen_y = int(p.y * tile_size) - cam_y
            size = int(p.size * (p.life / p.max_life))
            
            if size > 0:
                pygame.draw.circle(surface, (*p.color, alpha),
                                  (screen_x, screen_y), size)


# ==========================================
# EFFECT MANAGER
# ==========================================

class EffectManager:
    """
    Manages all visual effects with centralized update and render.
    
    Usage:
        effects = EffectManager()
        effects.add_pop((5, 10), color=(255, 220, 100))
        
        # In game loop:
        effects.update(dt)
        effects.render(surface, tile_size, camera_offset)
    """
    
    def __init__(self):
        self.effects: List[BaseEffect] = []
        self.trail: Optional[TrailEffect] = None
    
    def add_effect(self, effect: BaseEffect):
        """Add an effect to be managed."""
        self.effects.append(effect)
    
    def add_pop(self, position: Tuple[int, int], 
                color: Tuple[int, int, int] = (255, 220, 100)):
        """Convenience method to add a pop effect."""
        self.effects.append(PopEffect(position, color))
    
    def add_flash(self, position: Tuple[int, int],
                  color: Tuple[int, int, int] = (255, 255, 255)):
        """Convenience method to add a flash effect."""
        self.effects.append(FlashEffect(position, color))
    
    def add_ripple(self, position: Tuple[int, int],
                   color: Tuple[int, int, int] = (100, 200, 255)):
        """Convenience method to add a ripple effect."""
        self.effects.append(RippleEffect(position, color))
    
    def add_particles(self, position: Tuple[int, int],
                      color: Tuple[int, int, int] = (255, 220, 100),
                      count: int = 20):
        """Convenience method to add particle burst."""
        self.effects.append(ParticleEffect(position, color, count))
    
    def create_trail(self, color: Tuple[int, int, int] = (100, 150, 255)) -> TrailEffect:
        """Create and return a managed trail effect."""
        self.trail = TrailEffect(color)
        return self.trail
    
    def update(self, dt: float):
        """Update all effects."""
        # Update regular effects
        remaining = []
        for effect in self.effects:
            effect.update(dt)
            if effect.is_active():
                remaining.append(effect)
        self.effects = remaining
        
        # Update trail
        if self.trail:
            self.trail.update(dt)
    
    def render(self, surface: Surface, tile_size: int,
               camera_offset: Tuple[int, int] = (0, 0)):
        """Render all effects."""
        # Render trail first (behind other effects)
        if self.trail:
            self.trail.render(surface, tile_size, camera_offset)
        
        # Render other effects
        for effect in self.effects:
            effect.render(surface, tile_size, camera_offset)
    
    def clear(self):
        """Clear all effects."""
        self.effects.clear()
        if self.trail:
            self.trail.clear()


# ==========================================
# ITEM COLLECTION EFFECT
# ==========================================

class ItemCollectionEffect(BaseEffect):
    """
    Visual effect for collecting items (keys, bombs, etc.).
    
    Features:
    - Particle burst
    - Floating text message
    - Fade out over time
    """
    
    def __init__(self, pos: Tuple[int, int], item_type: str, 
                 icon: str = "🔑", message: str = "Key collected!"):
        self.position = pos
        self.item_type = item_type
        self.icon = icon
        self.message = message
        self.lifetime = 3.0
        self.elapsed = 0.0
        
        # Particle system
        self.particles: List[Tuple[float, float, float, float]] = []  # (x, y, vx, vy)
        for _ in range(15):
            import random
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(20, 50)
            self.particles.append((
                pos[1] + 0.5, pos[0] + 0.5,  # Start at center
                math.cos(angle) * speed,
                math.sin(angle) * speed
            ))
        
        # Colors based on item type
        self.colors = {
            'key': (255, 215, 0),
            'bomb': (255, 140, 0),
            'boss_key': (200, 50, 200),
            'triforce': (0, 255, 100)
        }
        self.color = self.colors.get(item_type, (255, 255, 255))
        
        # Font for floating text
        if PYGAME_AVAILABLE:
            self.font = pygame.font.SysFont('Arial', 14, bold=True)
    
    def update(self, dt: float) -> None:
        """Update particles and age."""
        self.elapsed += dt
        
        # Update particle positions
        new_particles = []
        for px, py, vx, vy in self.particles:
            # Apply velocity
            px += vx * dt
            py += vy * dt
            # Apply gravity (slow down)
            vy += 50 * dt
            vx *= 0.95
            new_particles.append((px, py, vx, vy))
        self.particles = new_particles
    
    def render(self, surface: Surface, tile_size: int,
               camera_offset: Tuple[int, int] = (0, 0)) -> None:
        """Render particles and floating text."""
        if not PYGAME_AVAILABLE:
            return
        
        progress = self.progress()
        alpha = int(255 * (1 - progress))
        
        # Render particles
        for px, py, vx, vy in self.particles:
            screen_x = int(px * tile_size - camera_offset[0])
            screen_y = int(py * tile_size - camera_offset[1])
            
            # Particle size decreases over time
            size = max(2, int(4 * (1 - progress)))
            
            # Create surface with alpha
            particle_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            color_with_alpha = (*self.color, alpha)
            pygame.draw.circle(particle_surf, color_with_alpha, (size, size), size)
            surface.blit(particle_surf, (screen_x - size, screen_y - size))
        
        # Render floating text (rises up)
        if self.font:
            text_y_offset = -progress * 30  # Float upward
            screen_x = int((self.position[1] + 0.5) * tile_size - camera_offset[0])
            screen_y = int((self.position[0] + 0.5) * tile_size - camera_offset[1] + text_y_offset)
            
            # Render icon
            icon_surf = self.font.render(self.icon, True, (*self.color, alpha))
            icon_rect = icon_surf.get_rect(center=(screen_x, screen_y - 20))
            
            # Add glow effect
            glow_surf = pygame.Surface(icon_surf.get_size(), pygame.SRCALPHA)
            glow_color = (*self.color, alpha // 2)
            glow_text = self.font.render(self.icon, True, glow_color)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx != 0 or dy != 0:
                        surface.blit(glow_text, (icon_rect.x + dx, icon_rect.y + dy))
            
            surface.blit(icon_surf, icon_rect)


# ==========================================
# ITEM USAGE EFFECT
# ==========================================

class ItemUsageEffect(BaseEffect):
    """
    Visual effect for using items (unlocking doors, bombing walls).
    
    Features:
    - Beam from Link to target (for keys)
    - Explosion animation (for bombs)
    - Screen shake (for bombs)
    """
    
    def __init__(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], 
                 item_type: str):
        self.from_pos = from_pos
        self.to_pos = to_pos
        self.item_type = item_type
        self.lifetime = 1.0 if item_type == "key" else 0.8
        self.elapsed = 0.0
        
        # Colors
        self.colors = {
            'key': (255, 215, 0),
            'bomb': (255, 100, 0),
            'boss_key': (200, 50, 200)
        }
        self.color = self.colors.get(item_type, (255, 255, 255))
        
        # Explosion particles for bombs
        self.explosion_particles: List[Tuple[float, float, float, float]] = []
        if item_type == "bomb":
            import random
            for _ in range(30):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(50, 150)
                self.explosion_particles.append((
                    to_pos[1] + 0.5, to_pos[0] + 0.5,
                    math.cos(angle) * speed,
                    math.sin(angle) * speed
                ))
    
    def update(self, dt: float) -> None:
        """Update effect."""
        self.elapsed += dt
        
        # Update explosion particles
        if self.item_type == "bomb":
            new_particles = []
            for px, py, vx, vy in self.explosion_particles:
                px += vx * dt
                py += vy * dt
                vx *= 0.92
                vy *= 0.92
                new_particles.append((px, py, vx, vy))
            self.explosion_particles = new_particles
    
    def render(self, surface: Surface, tile_size: int,
               camera_offset: Tuple[int, int] = (0, 0)) -> None:
        """Render unlock beam or explosion."""
        if not PYGAME_AVAILABLE:
            return
        
        progress = self.progress()
        
        if self.item_type == "key" or self.item_type == "boss_key":
            # Draw beam from Link to door
            from_x = int((self.from_pos[1] + 0.5) * tile_size - camera_offset[0])
            from_y = int((self.from_pos[0] + 0.5) * tile_size - camera_offset[1])
            to_x = int((self.to_pos[1] + 0.5) * tile_size - camera_offset[0])
            to_y = int((self.to_pos[0] + 0.5) * tile_size - camera_offset[1])
            
            # Animated beam (pulses)
            pulse = (math.sin(self.elapsed * 15) + 1) / 2
            alpha = int(255 * (1 - progress) * (0.5 + 0.5 * pulse))
            
            # Draw glow lines
            beam_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            for width in [8, 6, 4, 2]:
                beam_alpha = alpha // (9 - width)
                beam_color = (*self.color, beam_alpha)
                pygame.draw.line(beam_surf, beam_color, (from_x, from_y), (to_x, to_y), width)
            surface.blit(beam_surf, (0, 0))
            
            # Sparkles along beam
            num_sparkles = 5
            for i in range(num_sparkles):
                t = i / num_sparkles
                sparkle_x = int(from_x + (to_x - from_x) * t)
                sparkle_y = int(from_y + (to_y - from_y) * t)
                sparkle_size = int(3 * (1 - progress))
                if sparkle_size > 0:
                    sparkle_color = (*self.color, alpha)
                    pygame.draw.circle(beam_surf, sparkle_color, (sparkle_x, sparkle_y), sparkle_size)
        
        elif self.item_type == "bomb":
            # Draw explosion particles
            for px, py, vx, vy in self.explosion_particles:
                screen_x = int(px * tile_size - camera_offset[0])
                screen_y = int(py * tile_size - camera_offset[1])
                
                # Particle color shifts from yellow to red to black
                if progress < 0.3:
                    color = (255, 255, 100)
                elif progress < 0.6:
                    color = (255, 100, 0)
                else:
                    color = (100, 0, 0)
                
                alpha = int(255 * (1 - progress))
                size = max(1, int(5 * (1 - progress)))
                
                particle_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, (*color, alpha), (size, size), size)
                surface.blit(particle_surf, (screen_x - size, screen_y - size))
            
            # Central flash
            if progress < 0.4:
                flash_alpha = int(255 * (1 - progress / 0.4))
                flash_size = int(tile_size * 1.5 * (1 + progress))
                flash_x = int((self.to_pos[1] + 0.5) * tile_size - camera_offset[0])
                flash_y = int((self.to_pos[0] + 0.5) * tile_size - camera_offset[1])
                
                flash_surf = pygame.Surface((flash_size * 2, flash_size * 2), pygame.SRCALPHA)
                pygame.draw.circle(flash_surf, (255, 200, 0, flash_alpha), 
                                 (flash_size, flash_size), flash_size)
                surface.blit(flash_surf, (flash_x - flash_size, flash_y - flash_size))


# ==========================================
# ITEM MARKER EFFECT
# ==========================================

class ItemMarkerEffect(BaseEffect):
    """
    Glowing marker showing item position on map.
    Shows before collection, pulses for visibility.
    """
    
    def __init__(self, pos: Tuple[int, int], item_type: str, icon: str = "🔑"):
        self.position = pos
        self.item_type = item_type
        self.icon = icon
        self.lifetime = float('inf')  # Persistent until collected
        self.elapsed = 0.0
        
        # Colors
        self.colors = {
            'key': (255, 215, 0),
            'bomb': (255, 140, 0),
            'boss_key': (200, 50, 200),
            'triforce': (0, 255, 100)
        }
        self.color = self.colors.get(item_type, (255, 255, 255))
        
        # Font
        if PYGAME_AVAILABLE:
            self.font = pygame.font.SysFont('Arial', 12, bold=True)
    
    def update(self, dt: float) -> None:
        """Update animation."""
        self.elapsed += dt
    
    def render(self, surface: Surface, tile_size: int,
               camera_offset: Tuple[int, int] = (0, 0)) -> None:
        """Render pulsing marker."""
        if not PYGAME_AVAILABLE:
            return
        
        # Pulse animation
        pulse = (math.sin(self.elapsed * 3) + 1) / 2  # 0 to 1
        alpha = int(150 + 105 * pulse)
        
        screen_x = int((self.position[1] + 0.5) * tile_size - camera_offset[0])
        screen_y = int((self.position[0] + 0.5) * tile_size - camera_offset[1])
        
        # Glow circle
        glow_size = int(tile_size * 0.6 * (0.8 + 0.2 * pulse))
        glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.color, alpha // 2), (glow_size, glow_size), glow_size)
        surface.blit(glow_surf, (screen_x - glow_size, screen_y - glow_size))
        
        # Icon
        if self.font:
            icon_surf = self.font.render(self.icon, True, (*self.color, alpha))
            icon_rect = icon_surf.get_rect(center=(screen_x, screen_y))
            surface.blit(icon_surf, icon_rect)


# ==========================================
# EXPORTS
# ==========================================

__all__ = [
    'EffectState',
    'BaseEffect',
    'PopEffect',
    'FlashEffect',
    'TrailEffect',
    'TrailPoint',
    'RippleEffect',
    'PulseEffect',
    'ParticleEffect',
    'Particle',
    'ItemCollectionEffect',
    'ItemUsageEffect',
    'ItemMarkerEffect',
    'EffectManager',
]
