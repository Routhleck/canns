#!/usr/bin/env python3
"""
Generate CANNs Four Core Application Scenarios Diagram
Customize the text in the 'scenarios' dictionary below
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================
# CUSTOMIZABLE SECTION - Modify text here
# ============================================

# Main title
MAIN_TITLE = 'CANNs: Four Core Application Scenarios'

# Color scheme for each scenario
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96E6A1']
step_colors = ['#FFE5E5', '#E0F7F5', '#E3F4F9', '#E8F8EA']

# Scenario definitions - MODIFY TEXT HERE
scenarios = [
    {
        'title': 'Scenario 1',
        'subtitle': 'CANN Modeling and Simulation',
        'x': 2.5,
        'color': colors[0],
        'bg': step_colors[0],
        'steps': [
            'Model\nBuilding',           # Step 1 (Input - Yellow)
            'Task Data\nGeneration',      # Step 2 (Process - Gray)
            'Simulation\nExperiment',     # Step 3 (Process - Gray)
            'Model\nAnalysis'             # Step 4 (Output - Green)
        ]
    },
    {
        'title': 'Scenario 2',
        'subtitle': 'Data Analysis',
        'x': 6.7,
        'color': colors[1],
        'bg': step_colors[1],
        'steps': [
            'Real/Virtual\nExp. Data',        # Step 1 (Input - Yellow)
            'Data\nAnalysis',                  # Step 2 (Process - Gray)
            'Attractor/\nDynamics Analysis',  # Step 3 (Process - Gray)
            'Results'                          # Step 4 (Output - Green)
        ]
    },
    {
        'title': 'Scenario 3',
        'subtitle': 'Brain-Inspired Learning',
        'x': 10.9,
        'color': colors[2],
        'bg': step_colors[2],
        'steps': [
            'Task\nDataset',              # Step 1 (Input - Yellow)
            'Brain-Inspired\nModeling',     # Step 2 (Process - Gray)
            'Brain-Inspired\nTraining',     # Step 3 (Process - Gray)
            'Evaluation'                  # Step 4 (Output - Green)
        ]
    },
    {
        'title': 'Scenario 4',
        'subtitle': 'End-to-End Pipeline',
        'x': 15.1,
        'color': colors[3],
        'bg': step_colors[3],
        'steps': [
            'Input\nConfig',              # Step 1 (Input - Yellow)
            'Pipeline\nOrchestration',    # Step 2 (Process - Gray)
            'Auto\nExecution',            # Step 3 (Process - Gray)
            'Output\nReports'             # Step 4 (Output - Green)
        ]
    }
]

# Bottom legend text
LEGEND_TEXT = 'Input (Yellow) → Processing (Gray) → Output (Green)'

# Output filename
OUTPUT_FILENAME = 'canns_scenarios_custom'

# ============================================
# FONT SIZE SETTINGS - Adjust as needed
# ============================================
FONT_MAIN_TITLE = 32
FONT_SCENARIO_TITLE = 20
FONT_SUBTITLE = 14
FONT_STEP_TEXT = 13
FONT_LEGEND = 14

# ============================================
# DRAWING CODE - Usually no need to modify
# ============================================

def draw_compact_box(ax, x, y, text, color, w=3.2, h=0.85):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle='round,pad=0.02',
                         facecolor=color, edgecolor='#2C3E50', linewidth=2, alpha=0.9)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=FONT_STEP_TEXT, fontweight='bold')

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color='#34495E', lw=2.8, mutation_scale=15))

def main():
    # Create figure with 16:9 aspect ratio (widened for spacing)
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Main title
    ax.text(9, 8.5, MAIN_TITLE, fontsize=FONT_MAIN_TITLE, ha='center', fontweight='bold')

    # Draw each scenario
    for i, s in enumerate(scenarios):
        # Background box
        bg_box = FancyBboxPatch((s['x'] - 1.9, 0.7), 3.8, 7.0,
                                boxstyle='round,pad=0.05',
                                facecolor=s['bg'], edgecolor=s['color'],
                                linewidth=3.5, alpha=0.3)
        ax.add_patch(bg_box)

        # Scenario title
        ax.text(s['x'], 7.3, s['title'], ha='center', fontsize=FONT_SCENARIO_TITLE,
                fontweight='bold', color=s['color'])
        ax.text(s['x'], 6.8, s['subtitle'], ha='center', fontsize=FONT_SUBTITLE,
                style='italic', color='#444444')

        # Process steps (vertical flow)
        y_positions = [5.7, 4.3, 2.9, 1.5]
        for j, (step, y) in enumerate(zip(s['steps'], y_positions)):
            # Color coding: Input (yellow), Process (gray), Output (green)
            if j == 0:
                box_color = '#FFF3CD'  # Input - yellow
            elif j == len(s['steps']) - 1:
                box_color = '#D4EDDA'  # Output - green
            else:
                box_color = '#E2E3E5'  # Process - gray

            draw_compact_box(ax, s['x'], y, step, box_color)

            # Draw arrow to next step
            if j < len(s['steps']) - 1:
                draw_arrow(ax, s['x'], y - 0.43, s['x'], y_positions[j+1] + 0.43)

    # Bottom legend
    ax.text(9, 0.25, LEGEND_TEXT,
            ha='center', fontsize=FONT_LEGEND, color='#444444',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8F9FA',
                     edgecolor='#DEE2E6', linewidth=1.5))

    plt.tight_layout()

    # Save outputs
    plt.savefig(f'{OUTPUT_FILENAME}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{OUTPUT_FILENAME}.pdf', bbox_inches='tight', facecolor='white')

    print(f'Diagram saved as: {OUTPUT_FILENAME}.png and {OUTPUT_FILENAME}.pdf')
    print('You can now modify the text in the scenarios dictionary and re-run this script.')

if __name__ == '__main__':
    main()
