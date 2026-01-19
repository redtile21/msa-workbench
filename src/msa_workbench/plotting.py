import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pandas as pd

from msa_workbench.engine.msa_engine import MSAResult


def get_variability_chart(result: MSAResult, ax: plt.Axes):
    """Generates the Variability Chart on the given Axes."""
    ax.clear()
    cfg = result.config
    dfv = result.chart_data.variability.copy()

    y_col = cfg.response_col
    part_col = cfg.part_col
    op_col = cfg.operator_col

    other_factors = [f for f in cfg.factor_cols if f not in (part_col, op_col) and f in dfv.columns]
    inst_col = other_factors[0] if other_factors else None

    sort_cols = [op_col]
    if inst_col:
        sort_cols.append(inst_col)
    sort_cols.append(part_col)

    df_sorted = dfv.sort_values(by=sort_cols)
    group_cols = sort_cols
    unique_groups = df_sorted[group_cols].drop_duplicates().reset_index(drop=True)
    unique_groups['x_pos'] = unique_groups.index

    df_plot = pd.merge(df_sorted, unique_groups, on=group_cols)

    grand_mean = dfv[y_col].mean()
    sigma_gage = 0.0
    for vc in result.var_components:
        if "Gage R&R" in vc.source:
            sigma_gage = vc.std_dev
            break

    ucl = grand_mean + 3 * sigma_gage
    lcl = grand_mean - 3 * sigma_gage

    ax.scatter(df_plot['x_pos'], df_plot[y_col], s=40, alpha=0.8, edgecolors='black', zorder=3)

    ax.axhline(grand_mean, color='green', linewidth=1.5, label='Mean', zorder=2)
    ax.axhline(ucl, color='red', linestyle='--', linewidth=1.5, label='UCL', zorder=2)
    ax.axhline(lcl, color='red', linestyle='--', linewidth=1.5, label='LCL', zorder=2)

    trans_annot = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(1.01, grand_mean, f"Grand Mean = {grand_mean:.2f}", transform=trans_annot,
            color='green', va='center', ha='left', fontsize=9, fontweight='bold')
    ax.text(1.01, ucl, f"UCL = {ucl:.2f}", transform=trans_annot,
            color='red', va='center', ha='left', fontsize=9, fontweight='bold')
    ax.text(1.01, lcl, f"LCL = {lcl:.2f}", transform=trans_annot,
            color='red', va='center', ha='left', fontsize=9, fontweight='bold')

    ax.set_ylabel(y_col)
    ax.set_title(f"Variability Chart: {y_col} by {', '.join(group_cols)}")
    ax.set_xticks([])
    ax.set_xlim(-0.5, len(unique_groups) - 0.5)
    ax.grid(True, axis='y', linestyle=':', alpha=0.5, zorder=1)

    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    trans_label = mtransforms.blended_transform_factory(ax.transAxes, ax.transAxes)

    y_offset_part = -0.1
    y_offset_inst = -0.2
    y_offset_op = -0.3 if inst_col else -0.2

    for i, row in unique_groups.iterrows():
        clean_part_label = str(row[part_col]).replace(',', '').replace('(', '').replace(')', '')
        ax.text(row['x_pos'], y_offset_part, clean_part_label, transform=trans,
                ha='center', va='top', fontsize=9, rotation=90)

    ax.text(-0.01, y_offset_part, part_col, transform=trans_label, ha='right', va='top', fontsize=10, fontweight='bold')

    if inst_col:
        grouped_inst = unique_groups.groupby([op_col, inst_col], sort=False)
        for (op_val, inst_val), group in grouped_inst:
            first = group['x_pos'].min()
            last = group['x_pos'].max()
            center = (first + last) / 2
            clean_inst_label = str(inst_val).replace('\'', '').replace('(', '').replace(')', '')
            ax.text(center, y_offset_inst, clean_inst_label, transform=trans,
                    ha='center', va='top', fontsize=10, fontweight='bold')
            if last < len(unique_groups) - 1:
                ax.axvline(x=last + 0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.text(-0.01, y_offset_inst, inst_col, transform=trans_label, ha='right', va='top', fontsize=10,
                fontweight='bold')

    grouped_op = unique_groups.groupby([op_col], sort=False)
    for op_val, group in grouped_op:
        first = group['x_pos'].min()
        last = group['x_pos'].max()
        center = (first + last) / 2
        clean_op_label = str(op_val).replace('\'', '').replace('(', '').replace(')', '')
        ax.text(center, y_offset_op, clean_op_label, transform=trans,
                ha='center', va='top', fontsize=12, fontweight='bold')
        if last < len(unique_groups) - 1:
            ax.axvline(x=last + 0.5, color='black', linestyle='-', linewidth=1.5)

    ax.text(-0.01, y_offset_op, op_col, transform=trans_label, ha='right', va='top', fontsize=10, fontweight='bold')
    ax.figure.subplots_adjust(bottom=0.5 if inst_col else 0.4, right=0.9)


def get_stddev_chart(result: MSAResult, ax: plt.Axes):
    """Generates the Standard Deviation Chart on the given Axes."""
    ax.clear()
    cfg = result.config
    df_raw = result.chart_data.variability.copy()

    y_col = cfg.response_col
    part_col = cfg.part_col
    op_col = cfg.operator_col

    other_factors = [f for f in cfg.factor_cols if f not in (part_col, op_col) and f in df_raw.columns]
    inst_col = other_factors[0] if other_factors else None

    df_raw[part_col] = df_raw[part_col].astype(str)
    df_raw[op_col] = df_raw[op_col].astype(str)
    if inst_col:
        df_raw[inst_col] = df_raw[inst_col].astype(str)

    group_cols = [op_col]
    if inst_col:
        group_cols.append(inst_col)
    group_cols.append(part_col)

    df_std = df_raw.groupby(group_cols)[y_col].std().reset_index()
    df_std.rename(columns={y_col: 'cell_std'}, inplace=True)

    df_counts = df_raw.groupby(group_cols)[y_col].count().reset_index()
    n_mean = df_counts[y_col].mean()

    df_sorted = df_std.sort_values(by=group_cols)
    unique_groups = df_sorted[group_cols].drop_duplicates().reset_index(drop=True)
    unique_groups['x_pos'] = unique_groups.index

    df_plot = pd.merge(df_sorted, unique_groups, on=group_cols)

    s_bar = df_plot['cell_std'].mean()
    n = int(round(n_mean))

    B3_map = {
        2: 0, 3: 0, 4: 0, 5: 0, 6: 0.030, 7: 0.118, 8: 0.185, 9: 0.239, 10: 0.284,
        11: 0.321, 12: 0.354, 13: 0.382, 14: 0.406, 15: 0.428
    }
    B4_map = {
        2: 3.267, 3: 2.568, 4: 2.266, 5: 2.089, 6: 1.970, 7: 1.882, 8: 1.815, 9: 1.761, 10: 1.716,
        11: 1.679, 12: 1.646, 13: 1.618, 14: 1.594, 15: 1.572
    }

    b3 = B3_map.get(n, 0)
    b4 = B4_map.get(n, 3.267)

    ucl = s_bar * b4
    lcl = s_bar * b3

    ax.plot(df_plot['x_pos'], df_plot['cell_std'], marker='o', linestyle='-', color='blue', alpha=0.8, label='Std Dev',
            zorder=3)
    ax.axhline(s_bar, color='green', linewidth=1.5, label='Mean', zorder=2)
    ax.axhline(ucl, color='red', linestyle='--', linewidth=1.5, label='UCL', zorder=2)
    ax.axhline(lcl, color='red', linestyle='--', linewidth=1.5, label='LCL', zorder=2)

    trans_annot = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(1.01, s_bar, f"Mean = {s_bar:.2f}", transform=trans_annot, color='green', va='center', ha='left',
            fontsize=9, fontweight='bold')
    ax.text(1.01, ucl, f"UCL = {ucl:.2f}", transform=trans_annot, color='red', va='center', ha='left', fontsize=9,
            fontweight='bold')
    ax.text(1.01, lcl, f"LCL = {lcl:.2f}", transform=trans_annot, color='red', va='center', ha='left', fontsize=9,
            fontweight='bold')

    ax.set_ylabel("Standard Deviation")
    ax.set_title(f"Standard Deviation Chart by {', '.join([c.replace('(', '').replace(')', '') for c in group_cols])}")
    ax.set_xticks([])
    ax.set_xlim(-0.5, len(unique_groups) - 0.5)
    ax.grid(True, axis='y', linestyle=':', alpha=0.5, zorder=1)

    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    trans_label = mtransforms.blended_transform_factory(ax.transAxes, ax.transAxes)

    y_offset_part = -0.1
    y_offset_inst = -0.2
    y_offset_op = -0.3 if inst_col else -0.2

    for i, row in unique_groups.iterrows():
        clean_part_label = str(row[part_col]).replace('\'', '').replace('(', '').replace(')', '')
        ax.text(row['x_pos'], y_offset_part, clean_part_label, transform=trans, ha='center', va='top', fontsize=9,
                rotation=90)
    ax.text(-0.01, y_offset_part, part_col, transform=trans_label, ha='right', va='top', fontsize=10, fontweight='bold')

    if inst_col:
        grouped_inst = unique_groups.groupby([op_col, inst_col], sort=False)
        for (op_val, inst_val), group in grouped_inst:
            first, last = group['x_pos'].min(), group['x_pos'].max()
            center = (first + last) / 2
            clean_inst_label = str(inst_val).replace('\'', '').replace('(', '').replace(')', '')
            ax.text(center, y_offset_inst, clean_inst_label, transform=trans, ha='center', va='top', fontsize=10,
                    fontweight='bold')
            if last < len(unique_groups) - 1:
                ax.axvline(x=last + 0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.text(-0.01, y_offset_inst, inst_col, transform=trans_label, ha='right', va='top', fontsize=10,
                fontweight='bold')

    grouped_op = unique_groups.groupby(op_col, sort=False)
    for op_val, group in grouped_op:
        first, last = group['x_pos'].min(), group['x_pos'].max()
        center = (first + last) / 2
        clean_op_label = str(op_val).replace('\'', '').replace('(', '').replace(')', '')
        ax.text(center, y_offset_op, clean_op_label, transform=trans, ha='center', va='top', fontsize=12, fontweight='bold')
        if last < len(unique_groups) - 1:
            ax.axvline(x=last + 0.5, color='black', linestyle='-', linewidth=1.5)
    ax.text(-0.01, y_offset_op, op_col, transform=trans_label, ha='right', va='top', fontsize=10, fontweight='bold')
    ax.figure.subplots_adjust(bottom=0.5 if inst_col else 0.4, right=0.9)
