"""Spatial-pretraining curriculum registry (release-frozen).

Three curricula back the paper:
  synthetic_obb          - synthetic axis-aligned boxes, MAIN PAPER curriculum
                           (both benchmark-best checkpoints warm-start from it).
                           Kept verbatim.
  synthetic_obb_distract - synthetic_obb + (1,4) synthetic OBB distractors.
                           Kept verbatim.
  scene_harvested        - scene-harvested real-asset curriculum, the appendix
                           'Scene-harvested objects' study row
                           (--real_object_assets_enable). The trained appendix
                           curriculum additionally contained 12 landmark-maze
                           navigation entries
                           (object_class_route_plan_maze_landmark_{1..4}turn_real
                           x3, task machinery not part of this release), so
                           this entry ships without them. Sampling the trained
                           appendix mix exactly requires those tasks to be
                           re-registered through the external plugin seam
                           (curriculum_task_registry.py). Note that this mix
                           is NOT the synthetic mix with real objects: it has
                           no grounding family (the trained appendix battery
                           descended from an internal line that had dropped
                           object_class_grounding_real), adds
                           object_class_size_real, and weights the metric and
                           navigation tasks differently. For "the synthetic
                           curriculum, real objects, nothing else changed" use
                           the switch below.

Real-object switch (--curriculum_real_objects True). Every synthetic task has a
real-asset twin that asks the SAME question with the referent named by class
instead of by an inline patch marker. REAL_OBJECT_SUBSTITUTIONS maps them one
to one and real_object_tasks() rewrites a task list entry for entry, so the
weights of the chosen curriculum carry over unchanged. box_floor_area_* has no
twin (a room is not an object in the asset bank) and passes through, as it
does in scene_harvested. object_class_size_real has no synthetic counterpart
and is therefore not part of a switched mix.

argument.py consumes CURRICULA, CURRENT_TASKS, CURRENT_OBB_ONLY and
real_object_tasks. CURRENT_CURRICULUM is pinned to synthetic_obb (the
synthetic main-paper default).
"""

CURRICULA = {}

CURRICULA['synthetic_obb'] = {
    "tasks": 'dist_box,dist_box,dist_box,dist_box,dist_box,dist_box,dist_box,dist_box,rel_dist_box,rel_dist_box,rel_dist_box,rel_dist_box,rel_dist_box,rel_dist_box,rel_dist_box,rel_dist_box,box_floor_area_nonrect_yaw_obb_only,box_floor_area_nonrect_yaw_obb_only,box_floor_area_nonrect_yaw_obb_only,box_floor_area_nonrect_yaw_obb_only,box_floor_area_nonrect_yaw_obb_only,box_floor_area_nonrect_yaw_obb_only,box_floor_area_nonrect_yaw_obb_only,box_floor_area_nonrect_yaw_obb_only,rel_dir_easy_box,rel_dir_easy_box,rel_dir_easy_box,rel_dir_medium_box,rel_dir_medium_box,rel_dir_medium_box,rel_dir_hard_box,rel_dir_hard_box,rel_dir_hard_box,rel_dir_4way_box,rel_dir_4way_box,rel_dir_4way_box,rel_dir_camera_easy,rel_dir_camera_easy,rel_dir_camera_easy,rel_dir_camera_medium,rel_dir_camera_medium,rel_dir_camera_medium,rel_dir_camera_hard,rel_dir_camera_hard,rel_dir_camera_hard,rel_dir_oclock_box,rel_dir_oclock_box,rel_dir_oclock_box,route_plan_1turn_box,route_plan_1turn_box,route_plan_1turn_box,route_plan_1turn_box,route_plan_1turn_box,route_plan_1turn_box,route_plan_2turn_box,route_plan_2turn_box,route_plan_2turn_box,route_plan_2turn_box,route_plan_2turn_box,route_plan_2turn_box,route_plan_3turn_box,route_plan_3turn_box,route_plan_3turn_box,route_plan_3turn_box,route_plan_3turn_box,route_plan_3turn_box,route_plan_4turn_box,route_plan_4turn_box,route_plan_4turn_box,route_plan_4turn_box,route_plan_4turn_box,route_plan_4turn_box,appearance_order,appearance_order,appearance_order,appearance_order,appearance_order,appearance_order,appearance_order,appearance_order,appearance_order,appearance_order,appearance_order,appearance_order,visibility_from_pose,visibility_from_pose,visibility_from_pose,visibility_from_pose,visibility_from_pose,visibility_from_pose,visibility_from_pose,visibility_from_pose,visibility_from_pose,visibility_from_pose,visibility_from_pose,visibility_from_pose,object_counting,object_counting,object_counting,object_counting,object_counting,object_counting,object_counting_parity_box,object_counting_parity_box,object_counting_parity_box,object_counting_parity_box,object_counting_parity_box,object_counting_parity_box,object_counting_mod3_box,object_counting_mod3_box,object_counting_mod3_box,object_counting_mod3_box,object_counting_mod3_box,object_counting_mod3_box,rel_dir_count_side_box,rel_dir_count_side_box,rel_dir_count_side_box,rel_dir_count_side_box,rel_dir_count_side_box,rel_dir_count_side_box,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw',
    "obb_only": 'dist_box,rel_dist_box,box_floor_area_nonrect_yaw_obb_only,rel_dir_easy_box,rel_dir_medium_box,rel_dir_hard_box,rel_dir_4way_box,rel_dir_camera_easy,rel_dir_camera_medium,rel_dir_camera_hard,rel_dir_oclock_box,route_plan_1turn_box,route_plan_2turn_box,route_plan_3turn_box,route_plan_4turn_box,appearance_order,visibility_from_pose,object_counting,object_counting_parity_box,object_counting_mod3_box,rel_dir_count_side_box,multi_box_grounding_yaw,box_floor_area_yaw_obb_only',
}

CURRICULA['synthetic_obb_distract'] = {
    "tasks": 'dist_box,dist_box,dist_box,dist_box,dist_box,dist_box,dist_box,dist_box,rel_dist_box,rel_dist_box,rel_dist_box,rel_dist_box,rel_dist_box,rel_dist_box,rel_dist_box,rel_dist_box,box_floor_area_nonrect_yaw_obb_only,box_floor_area_nonrect_yaw_obb_only,box_floor_area_nonrect_yaw_obb_only,box_floor_area_nonrect_yaw_obb_only,box_floor_area_nonrect_yaw_obb_only,box_floor_area_nonrect_yaw_obb_only,box_floor_area_nonrect_yaw_obb_only,box_floor_area_nonrect_yaw_obb_only,rel_dir_easy_box,rel_dir_easy_box,rel_dir_easy_box,rel_dir_medium_box,rel_dir_medium_box,rel_dir_medium_box,rel_dir_hard_box,rel_dir_hard_box,rel_dir_hard_box,rel_dir_4way_box,rel_dir_4way_box,rel_dir_4way_box,rel_dir_camera_easy,rel_dir_camera_easy,rel_dir_camera_easy,rel_dir_camera_medium,rel_dir_camera_medium,rel_dir_camera_medium,rel_dir_camera_hard,rel_dir_camera_hard,rel_dir_camera_hard,rel_dir_oclock_box,rel_dir_oclock_box,rel_dir_oclock_box,route_plan_1turn_box,route_plan_1turn_box,route_plan_1turn_box,route_plan_1turn_box,route_plan_1turn_box,route_plan_1turn_box,route_plan_2turn_box,route_plan_2turn_box,route_plan_2turn_box,route_plan_2turn_box,route_plan_2turn_box,route_plan_2turn_box,route_plan_3turn_box,route_plan_3turn_box,route_plan_3turn_box,route_plan_3turn_box,route_plan_3turn_box,route_plan_3turn_box,route_plan_4turn_box,route_plan_4turn_box,route_plan_4turn_box,route_plan_4turn_box,route_plan_4turn_box,route_plan_4turn_box,appearance_order,appearance_order,appearance_order,appearance_order,appearance_order,appearance_order,appearance_order,appearance_order,appearance_order,appearance_order,appearance_order,appearance_order,visibility_from_pose,visibility_from_pose,visibility_from_pose,visibility_from_pose,visibility_from_pose,visibility_from_pose,visibility_from_pose,visibility_from_pose,visibility_from_pose,visibility_from_pose,visibility_from_pose,visibility_from_pose,object_counting,object_counting,object_counting,object_counting,object_counting,object_counting,object_counting_parity_box,object_counting_parity_box,object_counting_parity_box,object_counting_parity_box,object_counting_parity_box,object_counting_parity_box,object_counting_mod3_box,object_counting_mod3_box,object_counting_mod3_box,object_counting_mod3_box,object_counting_mod3_box,object_counting_mod3_box,rel_dir_count_side_box,rel_dir_count_side_box,rel_dir_count_side_box,rel_dir_count_side_box,rel_dir_count_side_box,rel_dir_count_side_box,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw,multi_box_grounding_yaw',
    "obb_only": 'dist_box,rel_dist_box,box_floor_area_nonrect_yaw_obb_only,rel_dir_easy_box,rel_dir_medium_box,rel_dir_hard_box,rel_dir_4way_box,rel_dir_camera_easy,rel_dir_camera_medium,rel_dir_camera_hard,rel_dir_oclock_box,route_plan_1turn_box,route_plan_2turn_box,route_plan_3turn_box,route_plan_4turn_box,appearance_order,visibility_from_pose,object_counting,object_counting_parity_box,object_counting_mod3_box,rel_dir_count_side_box,multi_box_grounding_yaw,box_floor_area_yaw_obb_only',
    'distractors': (1, 4),
}

CURRICULA['scene_harvested'] = {
    "tasks": 'object_class_dist_real,object_class_dist_real,object_class_dist_real,object_class_dist_real,object_class_dist_real,object_class_dist_real,object_class_rel_dist_real,object_class_rel_dist_real,object_class_rel_dist_real,object_class_rel_dist_real,object_class_rel_dist_real,object_class_rel_dist_real,box_floor_area_nonrect_yaw_obb_only,box_floor_area_nonrect_yaw_obb_only,box_floor_area_nonrect_yaw_obb_only,box_floor_area_nonrect_yaw_obb_only,box_floor_area_nonrect_yaw_obb_only,box_floor_area_nonrect_yaw_obb_only,object_class_size_real,object_class_size_real,object_class_size_real,object_class_size_real,object_class_size_real,object_class_size_real,object_class_rel_dir_easy_real,object_class_rel_dir_easy_real,object_class_rel_dir_easy_real,object_class_rel_dir_medium_real,object_class_rel_dir_medium_real,object_class_rel_dir_medium_real,object_class_rel_dir_hard_real,object_class_rel_dir_hard_real,object_class_rel_dir_hard_real,object_class_rel_dir_4way_real,object_class_rel_dir_4way_real,object_class_rel_dir_4way_real,object_class_rel_dir_camera_easy_real,object_class_rel_dir_camera_easy_real,object_class_rel_dir_camera_easy_real,object_class_rel_dir_camera_medium_real,object_class_rel_dir_camera_medium_real,object_class_rel_dir_camera_medium_real,object_class_rel_dir_camera_hard_real,object_class_rel_dir_camera_hard_real,object_class_rel_dir_camera_hard_real,object_class_rel_dir_oclock_real,object_class_rel_dir_oclock_real,object_class_rel_dir_oclock_real,object_class_appearance_order_real,object_class_appearance_order_real,object_class_appearance_order_real,object_class_appearance_order_real,object_class_appearance_order_real,object_class_appearance_order_real,object_class_appearance_order_real,object_class_appearance_order_real,object_class_appearance_order_real,object_class_appearance_order_real,object_class_appearance_order_real,object_class_appearance_order_real,object_class_visibility_from_pose_real,object_class_visibility_from_pose_real,object_class_visibility_from_pose_real,object_class_visibility_from_pose_real,object_class_visibility_from_pose_real,object_class_visibility_from_pose_real,object_class_visibility_from_pose_real,object_class_visibility_from_pose_real,object_class_visibility_from_pose_real,object_class_visibility_from_pose_real,object_class_visibility_from_pose_real,object_class_visibility_from_pose_real,object_class_counting_real,object_class_counting_real,object_class_counting_real,object_class_counting_real,object_class_counting_real,object_class_counting_real,object_class_counting_parity_real,object_class_counting_parity_real,object_class_counting_parity_real,object_class_counting_parity_real,object_class_counting_parity_real,object_class_counting_parity_real,object_class_counting_mod3_real,object_class_counting_mod3_real,object_class_counting_mod3_real,object_class_counting_mod3_real,object_class_counting_mod3_real,object_class_counting_mod3_real,object_class_rel_dir_count_side_real,object_class_rel_dir_count_side_real,object_class_rel_dir_count_side_real,object_class_rel_dir_count_side_real,object_class_rel_dir_count_side_real,object_class_rel_dir_count_side_real,object_class_route_plan_1turn_real,object_class_route_plan_2turn_real,object_class_route_plan_3turn_real,object_class_route_plan_4turn_real,object_class_route_plan_1turn_real,object_class_route_plan_2turn_real,object_class_route_plan_3turn_real,object_class_route_plan_4turn_real,object_class_route_plan_1turn_real,object_class_route_plan_2turn_real,object_class_route_plan_3turn_real,object_class_route_plan_4turn_real',
    "obb_only": 'object_class_dist_real,object_class_dist_real,object_class_dist_real,object_class_dist_real,object_class_dist_real,object_class_dist_real,object_class_rel_dist_real,object_class_rel_dist_real,object_class_rel_dist_real,object_class_rel_dist_real,object_class_rel_dist_real,object_class_rel_dist_real,box_floor_area_nonrect_yaw_obb_only,box_floor_area_nonrect_yaw_obb_only,box_floor_area_nonrect_yaw_obb_only,box_floor_area_nonrect_yaw_obb_only,box_floor_area_nonrect_yaw_obb_only,box_floor_area_nonrect_yaw_obb_only,object_class_size_real,object_class_size_real,object_class_size_real,object_class_size_real,object_class_size_real,object_class_size_real,object_class_rel_dir_easy_real,object_class_rel_dir_easy_real,object_class_rel_dir_easy_real,object_class_rel_dir_medium_real,object_class_rel_dir_medium_real,object_class_rel_dir_medium_real,object_class_rel_dir_hard_real,object_class_rel_dir_hard_real,object_class_rel_dir_hard_real,object_class_rel_dir_4way_real,object_class_rel_dir_4way_real,object_class_rel_dir_4way_real,object_class_rel_dir_camera_easy_real,object_class_rel_dir_camera_easy_real,object_class_rel_dir_camera_easy_real,object_class_rel_dir_camera_medium_real,object_class_rel_dir_camera_medium_real,object_class_rel_dir_camera_medium_real,object_class_rel_dir_camera_hard_real,object_class_rel_dir_camera_hard_real,object_class_rel_dir_camera_hard_real,object_class_rel_dir_oclock_real,object_class_rel_dir_oclock_real,object_class_rel_dir_oclock_real,object_class_appearance_order_real,object_class_appearance_order_real,object_class_appearance_order_real,object_class_appearance_order_real,object_class_appearance_order_real,object_class_appearance_order_real,object_class_appearance_order_real,object_class_appearance_order_real,object_class_appearance_order_real,object_class_appearance_order_real,object_class_appearance_order_real,object_class_appearance_order_real,object_class_visibility_from_pose_real,object_class_visibility_from_pose_real,object_class_visibility_from_pose_real,object_class_visibility_from_pose_real,object_class_visibility_from_pose_real,object_class_visibility_from_pose_real,object_class_visibility_from_pose_real,object_class_visibility_from_pose_real,object_class_visibility_from_pose_real,object_class_visibility_from_pose_real,object_class_visibility_from_pose_real,object_class_visibility_from_pose_real,object_class_counting_real,object_class_counting_real,object_class_counting_real,object_class_counting_real,object_class_counting_real,object_class_counting_real,object_class_counting_parity_real,object_class_counting_parity_real,object_class_counting_parity_real,object_class_counting_parity_real,object_class_counting_parity_real,object_class_counting_parity_real,object_class_counting_mod3_real,object_class_counting_mod3_real,object_class_counting_mod3_real,object_class_counting_mod3_real,object_class_counting_mod3_real,object_class_counting_mod3_real,object_class_rel_dir_count_side_real,object_class_rel_dir_count_side_real,object_class_rel_dir_count_side_real,object_class_rel_dir_count_side_real,object_class_rel_dir_count_side_real,object_class_rel_dir_count_side_real,object_class_route_plan_1turn_real,object_class_route_plan_2turn_real,object_class_route_plan_3turn_real,object_class_route_plan_4turn_real,object_class_route_plan_1turn_real,object_class_route_plan_2turn_real,object_class_route_plan_3turn_real,object_class_route_plan_4turn_real,object_class_route_plan_1turn_real,object_class_route_plan_2turn_real,object_class_route_plan_3turn_real,object_class_route_plan_4turn_real',
    'distractors': (1, 4),
    'real_distractors': (0, 3),
    'appearance_uniform_t': True,
}

CURRENT_CURRICULUM = 'synthetic_obb'
CURRENT_TASKS = CURRICULA[CURRENT_CURRICULUM]["tasks"]
CURRENT_OBB_ONLY = CURRICULA[CURRENT_CURRICULUM]["obb_only"]


# Synthetic task -> real-asset twin. Same question, same answer format, same
# scorer (utils.metrics.curriculum_score keys on the task name family); the
# referent is a class name and the pasted object carries its own per-patch
# features instead of one stashed feature cloned over a box.
REAL_OBJECT_SUBSTITUTIONS = {
    # metric
    "dist_box":                   "object_class_dist_real",
    "rel_dist_box":               "object_class_rel_dist_real",
    # egocentric direction
    "rel_dir_easy_box":           "object_class_rel_dir_easy_real",
    "rel_dir_medium_box":         "object_class_rel_dir_medium_real",
    "rel_dir_hard_box":           "object_class_rel_dir_hard_real",
    "rel_dir_4way_box":           "object_class_rel_dir_4way_real",
    "rel_dir_camera_easy":        "object_class_rel_dir_camera_easy_real",
    "rel_dir_camera_medium":      "object_class_rel_dir_camera_medium_real",
    "rel_dir_camera_hard":        "object_class_rel_dir_camera_hard_real",
    "rel_dir_oclock_box":         "object_class_rel_dir_oclock_real",
    # navigation
    "route_plan_1turn_box":       "object_class_route_plan_1turn_real",
    "route_plan_2turn_box":       "object_class_route_plan_2turn_real",
    "route_plan_3turn_box":       "object_class_route_plan_3turn_real",
    "route_plan_4turn_box":       "object_class_route_plan_4turn_real",
    # observability
    "appearance_order":           "object_class_appearance_order_real",
    "visibility_from_pose":       "object_class_visibility_from_pose_real",
    # counting
    "object_counting":            "object_class_counting_real",
    "object_counting_parity_box": "object_class_counting_parity_real",
    "object_counting_mod3_box":   "object_class_counting_mod3_real",
    "rel_dir_count_side_box":     "object_class_rel_dir_count_side_real",
    # grounding
    "multi_box_grounding_yaw":    "object_class_grounding_real",
}


def real_object_tasks(tasks_str):
    """Rewrite a comma-separated task list onto its real-object twins, entry
    for entry, so task weights are preserved. Tasks that are already real
    (``*_real``) and the floor-area tasks pass through. Any other task is an
    error rather than a silent synthetic leftover, because a marker task
    inside a "real objects" run would defeat the switch."""
    out = []
    unmapped = []
    for task in tasks_str.split(","):
        task = task.strip()
        if not task:
            continue
        if task in REAL_OBJECT_SUBSTITUTIONS:
            out.append(REAL_OBJECT_SUBSTITUTIONS[task])
        elif task.endswith("_real") or "floor_area" in task:
            out.append(task)
        else:
            unmapped.append(task)
    if unmapped:
        raise ValueError(
            "curriculum_real_objects: no real-object twin for "
            f"{sorted(set(unmapped))}. Add it to "
            "onecanvas.data.curriculum_task_mix.REAL_OBJECT_SUBSTITUTIONS "
            "or drop the task from the mix."
        )
    return ",".join(out)
