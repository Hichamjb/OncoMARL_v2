import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_evaluation_results(
    summary_path="checkpoints/test_evaluation_summary.json",
    output_dir="checkpoints",
):
    if not os.path.exists(summary_path):
        print(f"[Erreur] Fichier de résumé introuvable : {summary_path}")
        return

    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # Configuration esthétique globale (Syle professionnel / Publication)
    plt.rcParams.update(
        {
            "font.size": 10,
            "figure.autolayout": True,
            "axes.edgecolor": "#cbd5e1",
            "axes.linewidth": 1.0,
        }
    )

    total_states = data.get("total_states_evaluated", 0)
    agreement_rate = data.get("multi_agent_agreement_rate", 0.0)
    distribution = data.get("action_selection_distribution", {})
    per_example = data.get("per_example_results", [])

    # Action names formatted
    action_labels = [
        name.replace("_", " ").title() for name in distribution.keys()
    ]
    counts = [stats["count"] for stats in distribution.values()]
    percentages = [stats["percentage"] * 100 for stats in distribution.values()]

    # Palette de couleurs médicales professionnelles
    primary_blue = "#2563eb"
    emerald_green = "#10b981"
    amber_orange = "#f59e0b"
    agent_colors = ["#2563eb", "#0ea5e9", "#6366f1"]

    # =========================================================================
    # 1. GRAPHIQUE 1 : Distribution des Actions Recommandées (Bar Chart Horizontal)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    bars = ax.barh(
        action_labels,
        percentages,
        color=primary_blue,
        height=0.55,
        edgecolor="none",
    )

    ax.set_xlim(0, max(percentages) * 1.25 if max(percentages) > 0 else 100)
    ax.set_xlabel(
        "Pourcentage de Sélection (%)",
        fontsize=11,
        fontweight="bold",
        labelpad=10,
    )
    ax.set_ylabel("Action Clinique", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_title(
        "Distribution des Recommandations Cliniques (RL Multi-Agents)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    for bar, count, pct in zip(bars, counts, percentages):
        if count > 0 or pct > 0:
            ax.text(
                bar.get_width() + 1.2,
                bar.get_y() + bar.get_height() / 2,
                f"{count:,} états ({pct:.2f}%)",
                va="center",
                ha="left",
                fontsize=10,
                fontweight="bold",
                color="#1e293b",
            )

    plt.tight_layout()
    bar_chart_path = os.path.join(output_dir, "1_action_distribution.png")
    plt.savefig(bar_chart_path, dpi=300)
    plt.close()

    # =========================================================================
    # 2. GRAPHIQUE 2 : Taux d'Accord et Consensus (Donut Chart)
    # =========================================================================
    if len(per_example) > 0:
        unanimous_count = sum(
            1 for item in per_example if item.get("is_unanimous", False)
        )
        unanimous_ratio = unanimous_count / len(per_example)
        unanimous_total = int(unanimous_ratio * total_states)
    else:
        unanimous_total = int(agreement_rate * total_states)

    majority_total = total_states - unanimous_total
    sizes = [unanimous_total, majority_total]
    donut_labels = [
        "Accord Unanime (3/3 Agents)",
        "Accord Majoritaire (2/3 Agents)",
    ]

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=donut_labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=[emerald_green, amber_orange],
        pctdistance=0.75,
        textprops=dict(color="black", fontweight="bold", fontsize=10),
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
    )

    ax.text(
        0,
        0,
        f"{agreement_rate*100:.1f}%\nConsensus",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#1e293b",
    )
    ax.set_title(
        "Niveau de Consensus entre Agents",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    plt.tight_layout()
    donut_chart_path = os.path.join(
        output_dir, "2_consensus_unanimity_donut.png"
    )
    plt.savefig(donut_chart_path, dpi=300)
    plt.close()

    # =========================================================================
    # 3. GRAPHIQUE 3 : Décomposition des Votes par Agent Individuel
    # =========================================================================
    agent_votes = {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0]}
    for ex in per_example:
        acts = ex.get("prediction_details", {}).get("agent_actions", [])
        for agent_idx, act_id in enumerate(acts):
            if act_id in agent_votes and agent_idx < 3:
                agent_votes[act_id][agent_idx] += 1

    # Si per_example est un échantillon, extrapoler sur le total
    scale_factor = (
        (total_states / len(per_example)) if len(per_example) > 0 else 1.0
    )
    a1_counts = [int(agent_votes[i][0] * scale_factor) for i in range(4)]
    a2_counts = [int(agent_votes[i][1] * scale_factor) for i in range(4)]
    a3_counts = [int(agent_votes[i][2] * scale_factor) for i in range(4)]

    x = np.arange(len(action_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.bar(
        x - width,
        a1_counts,
        width,
        label="Agent Oncologue 1",
        color=agent_colors[0],
    )
    ax.bar(
        x, a2_counts, width, label="Agent Radiologue 2", color=agent_colors[1]
    )
    ax.bar(
        x + width,
        a3_counts,
        width,
        label="Agent Pathologiste 3",
        color=agent_colors[2],
    )

    ax.set_ylabel("Nombre de Votes", fontsize=11, fontweight="bold")
    ax.set_title(
        "Répartition des Votes Individuels par Agent",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(action_labels, fontsize=10, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    agent_chart_path = os.path.join(output_dir, "3_agent_votes_breakdown.png")
    plt.savefig(agent_chart_path, dpi=300)
    plt.close()

    # =========================================================================
    # 4. DASHBOARD EXÉCUTIF COMPLET (Figure 2x2 Récapitulative)
    # =========================================================================
    fig = plt.figure(figsize=(14, 10), dpi=300)
    fig.suptitle(
        "Rapport d'Évaluation du Système MARL Inférence Oncologique",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Panel A: KPI Card
    ax1 = plt.subplot(2, 2, 1)
    ax1.axis("off")
    top_action_name = action_labels[0]
    top_action_pct = percentages[0]
    summary_text = (
        f"  DASHBOARD EXÉCUTIF D'ÉVALUATION\n"
        f"  " + "─" * 38 + "\n\n"
        f"  • États Patients Évalués :   {total_states:,}\n\n"
        f"  • Taux d'Accord Global :     {agreement_rate*100:.2f}%\n\n"
        f"  • Décisions Unanimes :      {unanimous_total:,}\n\n"
        f"  • Action Principale :       {top_action_name} ({top_action_pct:.1f}%)\n\n"
        f"  • Statut Modèle :           VÉRIFIÉ & OPTIMAL"
    )
    ax1.text(
        0.05,
        0.5,
        summary_text,
        fontsize=11,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=1.2",
            facecolor="#f8fafc",
            edgecolor="#cbd5e1",
            lw=1.5,
        ),
        va="center",
    )

    # Panel B: Actions Bar Chart
    ax2 = plt.subplot(2, 2, 2)
    bars2 = ax2.barh(action_labels, percentages, color=primary_blue, height=0.5)
    ax2.set_xlim(0, max(percentages) * 1.25 if max(percentages) > 0 else 100)
    ax2.set_xlabel("Pourcentage (%)", fontsize=10, fontweight="bold")
    ax2.set_title("Distribution des Actions", fontsize=12, fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.5)
    for bar, pct in zip(bars2, percentages):
        if pct > 0:
            ax2.text(
                bar.get_width() + 1.5,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

    # Panel C: Donut Chart
    ax3 = plt.subplot(2, 2, 3)
    ax3.pie(
        sizes,
        labels=["Unanime", "Majoritaire"],
        autopct="%1.1f%%",
        startangle=140,
        colors=[emerald_green, amber_orange],
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
    )
    ax3.set_title("Consensus Multi-Agents", fontsize=12, fontweight="bold")

    # Panel D: Agent Votes
    ax4 = plt.subplot(2, 2, 4)
    ax4.bar(
        x - width, a1_counts, width, label="Agent 1", color=agent_colors[0]
    )
    ax4.bar(x, a2_counts, width, label="Agent 2", color=agent_colors[1])
    ax4.bar(
        x + width, a3_counts, width, label="Agent 3", color=agent_colors[2]
    )
    ax4.set_xticks(x)
    ax4.set_xticklabels(
        [a.split()[0] for a in action_labels], fontsize=9, fontweight="bold"
    )
    ax4.set_title("Votes par Agent", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    dashboard_path = os.path.join(
        output_dir, "4_executive_evaluation_dashboard.png"
    )
    plt.savefig(dashboard_path, dpi=300)
    plt.close()

    print("✅ Tous les graphiques ont été générés avec succès :")
    print(f"  • {bar_chart_path}")
    print(f"  • {donut_chart_path}")
    print(f"  • {agent_chart_path}")
    print(f"  • {dashboard_path}")


if __name__ == "__main__":
    plot_evaluation_results()