import { apiFetch, requireAuth } from "./auth.js";
import { VYRONEX_ACTIVITIES } from "./activities.js";

const dashboard = document.querySelector("[data-dashboard]");

if (dashboard) {
  const category = dashboard.dataset.dashboard;
  const activitiesContainer = dashboard.querySelector("[data-activities]");
  const errorMessage = dashboard.querySelector("[data-dashboard-error]");
  const statsContainer = dashboard.querySelector("[data-quick-stats]");

  async function initializeDashboard() {
    const user = await requireAuth(category, dashboard);
    if (!user) return;

    const name = dashboard.querySelector("[data-user-name]");
    if (name) name.textContent = user.name;

    const badge = dashboard.querySelector("[data-role-badge]");
    if (badge) badge.textContent = user.role === "gym" ? "Gym" : "Athlete";

    try {
      const response = await apiFetch(`/api/${category}/activities`);
      const result = await response.json();
      const allowedIds = new Set(result.activities);
      const activities = VYRONEX_ACTIVITIES.filter(
        (activity) => activity.category === category && allowedIds.has(activity.id),
      );
      if (activities.length !== allowedIds.size) {
        throw new Error("The activity catalog does not match this dashboard.");
      }

      renderActivities(activities);
      renderQuickStats(activities.length);
    } catch (error) {
      if (errorMessage) {
        errorMessage.textContent = error.message;
        errorMessage.hidden = false;
      }
    }
  }

  function renderActivities(activities) {
    activitiesContainer.replaceChildren();
    for (const activity of activities) {
      const card = document.createElement("a");
      card.className = "activity-card";
      card.href = `activity.html?activity=${encodeURIComponent(activity.id)}`;

      const categoryLabel = document.createElement("span");
      categoryLabel.className = "eyebrow";
      categoryLabel.textContent = category === "gym" ? "Gym training" : "Athletics";

      const title = document.createElement("h3");
      title.textContent = activity.label;

      const instructions = document.createElement("p");
      instructions.textContent = activity.instructions;

      const metrics = document.createElement("span");
      metrics.className = "activity-metrics";
      metrics.textContent = activity.metrics.join(" · ");

      card.append(categoryLabel, title, instructions, metrics);
      activitiesContainer.append(card);
    }
  }

  function renderQuickStats(activityCount) {
    const stats = [
      ["Sessions", "—"],
      ["Activities", String(activityCount)],
      ["Average form", "—"],
      ["Training time", "—"],
    ];

    statsContainer.replaceChildren();
    for (const [label, value] of stats) {
      const stat = document.createElement("div");
      stat.className = "quick-stat";
      const amount = document.createElement("strong");
      amount.textContent = value;
      const caption = document.createElement("span");
      caption.textContent = label;
      stat.append(amount, caption);
      statsContainer.append(stat);
    }
  }

  initializeDashboard();
}
