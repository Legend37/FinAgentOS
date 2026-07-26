# FinAgent OS 文档

运行入口与历史说明已分开存放，避免把设计稿误当成可部署页面：

- 应用前端：[`../fin_asset_agent/index.html`](../fin_asset_agent/index.html)
- 迭代记录：
  - [`2026-05-28 功能盘点`](iterations/2026-05-28-feature-overview.html)
  - [`2026-05-29 Agent 能力升级`](iterations/2026-05-29-agent-upgrades.html)
  - [`2026-06-02 持久 Agent 对话`](iterations/2026-06-02-persistent-chat.html)
  - [`2026-06-03 LLM 接入与持仓复盘`](iterations/2026-06-03-llm-and-review.html)
- 路线图：[`持久 Agent 路线图`](roadmap/persistent-agent-roadmap.html)

这些 HTML 是静态说明页，会从 Tailwind CDN 和 Google Fonts 加载样式。生产应用由 FastAPI 在 `/` 路径提供 `fin_asset_agent/index.html`，不依赖这些说明页。
