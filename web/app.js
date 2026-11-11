/**
 * PolicyMind 前端交互：
 * - fetch + ReadableStream 手动解析 SSE（EventSource 不支持 POST）
 * - 流式渲染答案、引用高亮，并驱动右侧多 Agent 链路可视化
 */
(() => {
    "use strict";

    const SESSION_KEY = "policymind_session_id";
    const API_STREAM = "/api/chat/stream";

    const AGENT_LABELS = {
        supervisor: "调度中枢",
        policy_qa: "制度问答",
        compare: "版本对比",
        eligibility: "资格计算",
        synthesize: "综合生成",
        critic: "质量核查",
    };

    const $ = (id) => document.getElementById(id);

    const els = {
        messages: $("messages"),
        welcome: $("welcome"),
        input: $("questionInput"),
        sendBtn: $("sendBtn"),
        intentBadge: $("intentBadge"),
        traceStatus: $("traceStatus"),
        reviewBanner: $("reviewBanner"),
        doneMeta: $("doneMeta"),
        citationList: $("citationList"),
        citationCount: $("citationCount"),
        modalOverlay: $("modalOverlay"),
        modalTitle: $("modalTitle"),
        modalMeta: $("modalMeta"),
        modalBody: $("modalBody"),
        modalClose: $("modalClose"),
    };

    const state = {
        busy: false,
        sessionId: getSessionId(),
        citations: [],
        rawAnswer: "",
        answer: null, // { content: HTMLElement, typing: HTMLElement | null }
        renderQueued: false,
        flashTimers: new Map(),
        running: new Set(),
        agentsSeen: [],
    };

    // ---------------------------------------------------------------- 会话

    function getSessionId() {
        try {
            let sid = localStorage.getItem(SESSION_KEY);
            if (!sid) {
                sid = createId();
                localStorage.setItem(SESSION_KEY, sid);
            }
            return sid;
        } catch (err) {
            return createId();
        }
    }

    function createId() {
        if (window.crypto && typeof window.crypto.randomUUID === "function") {
            return window.crypto.randomUUID();
        }
        return "s-" + Date.now() + "-" + Math.random().toString(16).slice(2, 10);
    }

    function collectScopes() {
        const checked = Array.from(document.querySelectorAll(".scope-chip input:checked"))
            .map((input) => input.value)
            .filter(Boolean);
        return checked.length ? checked : null;
    }

    // ---------------------------------------------------------------- 消息渲染

    function escapeHtml(text) {
        return String(text)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    function renderAnswer() {
        if (!state.answer) return;
        const text = state.rawAnswer || "";
        let html = escapeHtml(text).replace(/\n/g, "<br>");
        html = html.replace(/【E(\d+)】/g, (match, num) =>
            `<span class="cite-ref" data-ref="E${num}" role="button" tabindex="0">${match}</span>`
        );
        state.answer.content.innerHTML = html;

        state.answer.content.querySelectorAll(".cite-ref").forEach((node) => {
            const citation = findCitation(node.dataset.ref);
            if (citation) {
                const article = citation.articleNo ? `第${citation.articleNo}条` : "";
                node.title = `《${citation.title}》${article}`;
            }
        });
    }

    function scheduleRender() {
        if (state.renderQueued) return;
        state.renderQueued = true;
        window.requestAnimationFrame(() => {
            state.renderQueued = false;
            renderAnswer();
            scrollChat(false);
        });
    }

    function scrollChat(force) {
        const el = els.messages;
        if (!el) return;
        const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 140;
        if (force || nearBottom) {
            el.scrollTop = el.scrollHeight;
        }
    }

    function appendUserMessage(text) {
        const wrap = document.createElement("div");
        wrap.className = "message user";

        const bubble = document.createElement("div");
        bubble.className = "bubble";

        const role = document.createElement("div");
        role.className = "msg-role";
        role.textContent = "我";

        const content = document.createElement("div");
        content.className = "user-content";
        content.textContent = text;

        bubble.append(role, content);
        wrap.appendChild(bubble);
        els.messages.appendChild(wrap);
        scrollChat(true);
    }

    function createAssistantMessage() {
        const wrap = document.createElement("div");
        wrap.className = "message assistant";

        const bubble = document.createElement("div");
        bubble.className = "bubble";

        const role = document.createElement("div");
        role.className = "msg-role";
        role.textContent = "PolicyMind";

        const content = document.createElement("div");
        content.className = "answer-content streaming";

        const typing = document.createElement("div");
        typing.className = "typing-indicator";
        typing.setAttribute("aria-label", "正在思考");
        typing.innerHTML = "<span></span><span></span><span></span>";

        bubble.append(role, content, typing);
        wrap.appendChild(bubble);
        els.messages.appendChild(wrap);

        state.answer = { content, typing };
        scrollChat(true);
        return state.answer;
    }

    function removeTyping() {
        if (state.answer && state.answer.typing) {
            state.answer.typing.remove();
            state.answer.typing = null;
        }
    }

    function stopStreaming() {
        if (!state.answer) return;
        state.answer.content.classList.remove("streaming");
        removeTyping();
    }

    function showStreamError(message) {
        const text = message ? String(message) : "未知错误";
        if (!state.answer) return;
        removeTyping();
        state.rawAnswer += (state.rawAnswer ? "\n\n" : "") + "【错误】" + text;
        renderAnswer();
        state.answer.content.classList.add("has-error");
        state.answer.content.classList.remove("streaming");
        scrollChat(true);
        setTraceStatus("发生错误：" + text, "error");
    }

    // ---------------------------------------------------------------- 引用

    function normalizeRefId(value) {
        if (value === null || value === undefined || value === "") return null;
        const text = String(value);
        return /^\d+$/.test(text) ? "E" + text : text;
    }

    function addCitation(data) {
        if (!data || typeof data !== "object") return;
        const refIdRaw = normalizeRefId(data.ref_id ?? data.refId ?? data.cite_id);
        const title = String(data.title || data.doc_title || "未命名制度");
        const articleNo = data.article_no ?? data.articleNo ?? data.article ?? "";
        const page = data.page ?? null;
        const snippet = String(data.snippet || data.text || data.content || "");
        const docId = String(data.doc_id || data.docId || "");

        let citation = null;
        if (refIdRaw) {
            citation = state.citations.find((item) => item.refId === refIdRaw) || null;
        }
        if (!citation && (docId || articleNo)) {
            citation = state.citations.find(
                (item) => item.docId === docId && String(item.articleNo) === String(articleNo)
            ) || null;
        }

        if (citation) {
            citation.title = title;
            citation.articleNo = articleNo;
            citation.page = page;
            citation.snippet = snippet || citation.snippet;
            citation.docId = docId || citation.docId;
        } else {
            citation = {
                refId: refIdRaw || "E" + (state.citations.length + 1),
                title,
                articleNo,
                page,
                snippet,
                docId,
            };
            state.citations.push(citation);
        }

        renderCitations();
        if (state.rawAnswer) renderAnswer();
    }

    function findCitation(refId) {
        return state.citations.find((item) => item.refId === refId) || null;
    }

    function renderCitations() {
        els.citationList.replaceChildren();
        els.citationCount.textContent = String(state.citations.length);

        if (!state.citations.length) {
            const empty = document.createElement("p");
            empty.className = "empty";
            empty.textContent = "暂无引用";
            els.citationList.appendChild(empty);
            return;
        }

        state.citations.forEach((citation) => {
            const card = document.createElement("button");
            card.type = "button";
            card.className = "citation-card";
            card.addEventListener("click", () => openCitation(citation));

            const ref = document.createElement("span");
            ref.className = "citation-ref";
            ref.textContent = citation.refId;

            const body = document.createElement("span");
            body.className = "citation-body";

            const title = document.createElement("span");
            title.className = "citation-title";
            title.textContent = "《" + citation.title + "》";

            const meta = document.createElement("span");
            meta.className = "citation-meta";
            const parts = [];
            if (citation.articleNo !== "" && citation.articleNo !== null) {
                parts.push("第 " + citation.articleNo + " 条");
            }
            if (citation.page !== null && citation.page !== "") {
                parts.push("第 " + citation.page + " 页");
            }
            meta.textContent = parts.join(" · ") || "查看条款详情";

            body.append(title, meta);
            card.append(ref, body);
            els.citationList.appendChild(card);
        });
    }

    function openCitation(citation) {
        els.modalTitle.textContent = "【" + citation.refId + "】《" + citation.title + "》";
        const parts = [];
        if (citation.articleNo !== "" && citation.articleNo !== null) {
            parts.push("第 " + citation.articleNo + " 条");
        }
        if (citation.page !== null && citation.page !== "") {
            parts.push("第 " + citation.page + " 页");
        }
        if (citation.docId) parts.push("doc: " + citation.docId);
        els.modalMeta.textContent = parts.join(" · ");
        els.modalBody.textContent = citation.snippet || "（该条款暂无摘要）";
        els.modalOverlay.classList.remove("hidden");
        els.modalClose.focus();
    }

    function closeModal() {
        els.modalOverlay.classList.add("hidden");
    }

    function handleCiteRefClick(refId) {
        const citation = findCitation(refId);
        if (citation) {
            openCitation(citation);
        } else {
            setTraceStatus("引用 " + refId + " 的详情尚未推送", "error");
        }
    }

    // ---------------------------------------------------------------- 链路面板

    function setTraceStatus(text, tone) {
        els.traceStatus.textContent = text;
        els.traceStatus.classList.toggle("error", tone === "error");
    }

    function nodeFor(agent) {
        return document.querySelector('.node[data-agent="' + agent + '"]');
    }

    function normalizeAgent(name) {
        const raw = String(name || "").toLowerCase().trim();
        if (!raw) return null;
        if (raw.indexOf("supervisor") !== -1 || raw.indexOf("route") !== -1 || raw.indexOf("dispatch") !== -1) {
            return "supervisor";
        }
        if (raw.indexOf("policy") !== -1 || raw.indexOf("qa") !== -1 || raw.indexOf("retriev") !== -1) {
            return "policy_qa";
        }
        if (raw.indexOf("compare") !== -1 || raw.indexOf("cmp") !== -1) return "compare";
        if (raw.indexOf("eligib") !== -1 || raw.indexOf("eli") !== -1) return "eligibility";
        if (raw.indexOf("synth") !== -1) return "synthesize";
        if (raw.indexOf("critic") !== -1 || raw.indexOf("review") !== -1) return "critic";
        return null;
    }

    function agentLabel(agent) {
        return AGENT_LABELS[agent] || String(agent || "Agent");
    }

    function setNodeRunning(agent, running) {
        if (!agent) return;
        const node = nodeFor(agent);
        if (!node) return;

        if (running) {
            state.running.add(agent);
            node.classList.add("running");
            const timer = state.flashTimers.get(agent);
            if (timer) {
                window.clearTimeout(timer);
                state.flashTimers.delete(agent);
            }
        } else {
            state.running.delete(agent);
            node.classList.remove("running");
        }
    }

    function flashNode(agent, duration) {
        const node = nodeFor(agent);
        if (!node) return;
        node.classList.add("running");
        const existing = state.flashTimers.get(agent);
        if (existing) window.clearTimeout(existing);
        const timer = window.setTimeout(() => {
            state.flashTimers.delete(agent);
            if (!state.running.has(agent)) node.classList.remove("running");
        }, duration || 900);
        state.flashTimers.set(agent, timer);
    }

    function setNodeTool(agent, text) {
        const node = nodeFor(agent);
        if (!node) return;
        const badge = node.querySelector(".node-tool");
        if (badge) badge.textContent = text;
    }

    function resetTrace() {
        state.citations = [];
        state.rawAnswer = "";
        state.answer = null;
        state.renderQueued = false;
        state.running.clear();
        state.agentsSeen = [];
        state.flashTimers.forEach((timer) => window.clearTimeout(timer));
        state.flashTimers.clear();

        document.querySelectorAll(".node").forEach((node) => {
            node.classList.remove("running");
            const badge = node.querySelector(".node-tool");
            if (badge) badge.textContent = "";
        });

        els.intentBadge.textContent = "意图：识别中…";
        els.intentBadge.classList.remove("active");
        els.intentBadge.removeAttribute("title");

        setTraceStatus("正在启动多 Agent 链路…");
        els.reviewBanner.className = "review-banner hidden";
        els.reviewBanner.replaceChildren();
        els.doneMeta.className = "done-meta hidden";
        els.doneMeta.replaceChildren();
        renderCitations();
    }

    function updateIntent(data) {
        const intentRaw = data.intent || data.route || data.intent_type || "unknown";
        const intent = typeof intentRaw === "string" ? intentRaw : stringifyValue(intentRaw);
        els.intentBadge.textContent = "意图：" + intent;
        els.intentBadge.classList.add("active");

        const plan = Array.isArray(data.plan) ? data.plan : [];
        if (plan.length) {
            const summary = plan
                .map((item) => {
                    if (typeof item === "string") return item;
                    if (item && typeof item === "object") {
                        return item.goal || item.agent || item.task || JSON.stringify(item);
                    }
                    return String(item);
                })
                .filter(Boolean)
                .join("；");
            if (summary) els.intentBadge.title = summary;
        }
        setTraceStatus("调度中枢识别意图：" + intent);
        flashNode("supervisor", 1100);
    }

    function showReview(data) {
        const bad = Array.isArray(data.bad_citations) ? data.bad_citations : [];
        const verdictRaw = data.verdict;
        const verdictText = typeof verdictRaw === "string"
            ? verdictRaw
            : (verdictRaw === null || verdictRaw === undefined ? "" : stringifyValue(verdictRaw));
        const lower = verdictText.toLowerCase();

        const failWords = ["reject", "fail", "不通过", "未通过", "驳回", "存疑"];
        const passWords = ["pass", "ok", "approve", "通过", "合格", "一致"];
        let passed;
        if (!lower) {
            passed = bad.length === 0;
        } else {
            passed = !failWords.some((word) => lower.indexOf(word) !== -1)
                && passWords.some((word) => lower.indexOf(word) !== -1);
        }

        els.reviewBanner.className = "review-banner " + (passed ? "pass" : "fail");
        els.reviewBanner.replaceChildren();

        const head = document.createElement("div");
        head.className = "review-head";

        const tag = document.createElement("span");
        tag.className = "review-tag";
        tag.textContent = passed ? "核查通过" : "核查未通过";
        head.appendChild(tag);

        if (verdictText) {
            const raw = document.createElement("span");
            raw.className = "review-verdict";
            raw.textContent = "verdict: " + verdictText;
            head.appendChild(raw);
        }
        els.reviewBanner.appendChild(head);

        if (bad.length) {
            const list = document.createElement("div");
            list.className = "review-bad";
            const label = document.createElement("span");
            label.className = "review-bad-label";
            label.textContent = "bad_citations：";
            list.appendChild(label);
            bad.forEach((item) => {
                const chip = document.createElement("span");
                chip.className = "review-chip";
                chip.textContent = stringifyValue(item);
                list.appendChild(chip);
            });
            els.reviewBanner.appendChild(list);
        }

        setTraceStatus(passed ? "质量核查通过" : "质量核查未通过，存在引用问题", passed ? "" : "error");
    }

    function showDone(data) {
        const agents = Array.isArray(data.agents_used) && data.agents_used.length
            ? data.agents_used.map((agent) => agentLabel(normalizeAgent(agent) || agent))
            : state.agentsSeen.map((agent) => agentLabel(normalizeAgent(agent) || agent));

        const items = [
            ["耗时", formatDuration(data.latency_ms)],
            ["Token", formatTokens(data.token_usage)],
            ["Agents", agents.length ? agents.join("、") : "—"],
        ];

        els.doneMeta.className = "done-meta";
        els.doneMeta.replaceChildren();
        items.forEach(([label, value]) => {
            const item = document.createElement("span");
            item.className = "done-item";

            const labelEl = document.createElement("span");
            labelEl.className = "done-label";
            labelEl.textContent = label;

            const valueEl = document.createElement("span");
            valueEl.className = "done-value";
            valueEl.textContent = value;

            item.append(labelEl, valueEl);
            els.doneMeta.appendChild(item);
        });
        els.doneMeta.classList.remove("hidden");
    }

    function handleDone(data) {
        stopStreaming();

        // 图会先把各专家的过程 token 推给前端，最终综合答案只在 done.final_answer 中；
        // 因此只要终稿非空就以终稿覆盖，避免界面停留在专家中间输出上。
        if (typeof data.final_answer === "string" && data.final_answer.trim()) {
            state.rawAnswer = data.final_answer;
            renderAnswer();
        }
        if (Array.isArray(data.citations)) {
            data.citations.forEach(addCitation);
        }

        state.running.forEach((agent) => setNodeRunning(agent, false));
        state.flashTimers.forEach((timer) => window.clearTimeout(timer));
        state.flashTimers.clear();
        document.querySelectorAll(".node.running").forEach((node) => node.classList.remove("running"));

        showDone(data);
        setTraceStatus("本轮链路处理完成");
        scrollChat(true);
    }

    // ---------------------------------------------------------------- 事件分发

    function handleEvent(eventName, data) {
        switch (eventName) {
            case "route":
                updateIntent(data);
                break;
            case "agent_start": {
                const agent = normalizeAgent(data.agent);
                if (data.agent && state.agentsSeen.indexOf(String(data.agent)) === -1) {
                    state.agentsSeen.push(String(data.agent));
                }
                if (agent) {
                    setNodeRunning(agent, true);
                    const goal = data.goal ? "：" + truncate(data.goal, 36) : "";
                    setTraceStatus(agentLabel(agent) + " 开始执行" + goal);
                } else {
                    setTraceStatus("Agent " + (data.agent || "") + " 开始执行");
                }
                break;
            }
            case "agent_end": {
                const agent = normalizeAgent(data.agent);
                if (data.agent && state.agentsSeen.indexOf(String(data.agent)) === -1) {
                    state.agentsSeen.push(String(data.agent));
                }
                if (agent) setNodeRunning(agent, false);
                const elapsed = data.elapsed_ms !== undefined && data.elapsed_ms !== null
                    ? " · " + formatDuration(data.elapsed_ms)
                    : "";
                setTraceStatus((agent ? agentLabel(agent) : (data.agent || "Agent")) + " 完成" + elapsed);
                break;
            }
            case "tool": {
                const agent = normalizeAgent(data.agent);
                const tool = data.tool || "tool";
                if (agent) {
                    const hits = formatHits(data.hits);
                    setNodeTool(agent, hits ? tool + " · " + hits : tool);
                }
                const hitsText = data.hits === undefined || data.hits === null
                    ? ""
                    : "（" + formatHits(data.hits) + " 条）";
                setTraceStatus((agent ? agentLabel(agent) : (data.agent || "Agent"))
                    + " 调用工具 " + tool + hitsText);
                break;
            }
            case "retrieve": {
                const count = data.count === undefined || data.count === null ? "—" : data.count;
                const elapsed = data.elapsed_ms === undefined || data.elapsed_ms === null
                    ? ""
                    : " · " + formatDuration(data.elapsed_ms);
                setTraceStatus("召回通道 " + (data.source || "unknown") + "：" + count + " 条" + elapsed);
                break;
            }
            case "grade": {
                const coverage = formatCoverage(data.coverage);
                const missing = Array.isArray(data.missing) && data.missing.length
                    ? "，缺口：" + data.missing.slice(0, 3).join("、")
                    : "";
                const loop = data.loop === undefined || data.loop === null
                    ? ""
                    : "（第 " + (Number(data.loop) + 1) + " 轮）";
                setTraceStatus("召回覆盖 " + coverage + missing + loop);
                break;
            }
            case "rerank": {
                const topK = data.top_k !== undefined && data.top_k !== null
                    ? data.top_k
                    : (Array.isArray(data.items) ? data.items.length : "—");
                setTraceStatus("精排保留 " + topK + " 条候选");
                break;
            }
            case "synthesize": {
                const evidence = data.evidence_count === undefined || data.evidence_count === null
                    ? "—"
                    : data.evidence_count;
                const cites = data.citations === undefined || data.citations === null
                    ? "—"
                    : data.citations;
                const fallback = data.fallback ? "（模板兜底）" : "";
                flashNode("synthesize", 900);
                setTraceStatus("综合生成完成：证据 " + evidence + " 条 · 引用 " + cites + " 条" + fallback);
                break;
            }
            case "clarify": {
                const degraded = data.degraded ? "（降级输出）" : "";
                flashNode("supervisor", 900);
                setTraceStatus("信息不足或制度未明确，输出澄清回答" + degraded);
                break;
            }
            case "review":
                showReview(data);
                break;
            case "token": {
                if (typeof data.text !== "string") break;
                state.rawAnswer += data.text;
                removeTyping();
                scheduleRender();
                break;
            }
            case "citation":
                addCitation(data);
                break;
            case "done":
                handleDone(data);
                break;
            case "error": {
                const message = data.message || data.error || data.detail || stringifyValue(data);
                showStreamError(message);
                break;
            }
            default:
                if (eventName && eventName !== "message") {
                    setTraceStatus("事件 " + eventName);
                }
        }
    }

    // ---------------------------------------------------------------- SSE 解析

    async function streamChat(question) {
        const response = await fetch(API_STREAM, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            body: JSON.stringify({
                session_id: state.sessionId,
                question: question,
                user_scope: collectScopes(),
            }),
        });

        if (!response.ok) {
            let detail = "HTTP " + response.status;
            try {
                const payload = await response.json();
                if (payload && payload.detail) {
                    detail = typeof payload.detail === "string"
                        ? payload.detail
                        : stringifyValue(payload.detail);
                }
            } catch (err) {
                /* 非 JSON 错误体，保留状态码提示 */
            }
            throw new Error(detail);
        }
        if (!response.body) {
            throw new Error("当前浏览器不支持流式响应");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        try {
            while (true) {
                const result = await reader.read();
                if (result.done) break;

                buffer += decoder.decode(result.value, { stream: true });
                buffer = buffer.replace(/\r\n/g, "\n");

                let boundary = buffer.indexOf("\n\n");
                while (boundary !== -1) {
                    const frame = buffer.slice(0, boundary);
                    buffer = buffer.slice(boundary + 2);
                    processFrame(frame);
                    boundary = buffer.indexOf("\n\n");
                }
            }

            buffer += decoder.decode();
            buffer = buffer.replace(/\r\n/g, "\n");
            if (buffer.trim()) processFrame(buffer);
        } finally {
            try {
                reader.releaseLock();
            } catch (err) {
                /* ignore */
            }
        }
    }

    function processFrame(frame) {
        if (!frame || !frame.trim()) return;

        let eventName = "message";
        const dataLines = [];

        frame.split("\n").forEach((line) => {
            if (!line || line.startsWith(":")) return;
            if (line.startsWith("event:")) {
                eventName = line.slice(6).trim();
            } else if (line.startsWith("data:")) {
                dataLines.push(line.slice(5).replace(/^ /, ""));
            }
        });

        // retry: / 注释帧没有 data，直接忽略。
        if (!dataLines.length) return;

        const payloadText = dataLines.join("\n");
        let data;
        try {
            data = JSON.parse(payloadText);
        } catch (err) {
            console.warn("SSE 数据解析失败：", payloadText, err);
            return;
        }
        handleEvent(eventName, data && typeof data === "object" ? data : { value: data });
    }

    // ---------------------------------------------------------------- 发送流程

    async function sendQuestion() {
        if (state.busy) return;

        const question = els.input.value.trim();
        if (!question) {
            els.input.focus();
            return;
        }

        els.input.value = "";
        autoGrow();
        if (els.welcome) els.welcome.classList.add("hidden");

        appendUserMessage(question);
        resetTrace();
        createAssistantMessage();
        setBusy(true);

        try {
            await streamChat(question);
        } catch (err) {
            const message = err && err.message ? err.message : "网络请求失败";
            showStreamError("请求失败：" + message);
        } finally {
            stopStreaming();
            state.running.forEach((agent) => setNodeRunning(agent, false));
            setBusy(false);
            els.input.focus();
        }
    }

    function setBusy(busy) {
        state.busy = busy;
        els.sendBtn.disabled = busy;
        els.sendBtn.classList.toggle("loading", busy);
        els.sendBtn.setAttribute("aria-busy", busy ? "true" : "false");
        const label = els.sendBtn.querySelector(".send-label");
        if (label) label.textContent = busy ? "思考中…" : "发送";
    }

    function autoGrow() {
        const el = els.input;
        el.style.height = "auto";
        el.style.height = Math.min(el.scrollHeight, 160) + "px";
    }

    // ---------------------------------------------------------------- 工具函数

    function truncate(text, max) {
        const value = String(text || "");
        return value.length > max ? value.slice(0, max) + "…" : value;
    }

    function formatHits(hits) {
        if (hits === null || hits === undefined || hits === "") return "";
        if (Array.isArray(hits)) return String(hits.length);
        return String(hits);
    }

    function formatCoverage(value) {
        const number = Number(value);
        if (!Number.isFinite(number)) return "—";
        const percent = number <= 1 ? number * 100 : number;
        return Math.round(percent) + "%";
    }

    function formatDuration(value) {
        const ms = Number(value);
        if (!Number.isFinite(ms)) return "—";
        if (ms < 1000) return Math.round(ms) + " ms";
        return (ms / 1000).toFixed(ms < 10000 ? 1 : 0) + " s";
    }

    function formatTokens(usage) {
        if (usage === null || usage === undefined) return "—";
        if (typeof usage === "number") return usage.toLocaleString("zh-CN");
        if (typeof usage !== "object") return String(usage);

        const totalTokens = Number(usage.total_tokens);
        if (usage.total_tokens !== undefined && usage.total_tokens !== null && Number.isFinite(totalTokens)) {
            return totalTokens.toLocaleString("zh-CN");
        }
        const total = Number(usage.total);
        if (usage.total !== undefined && usage.total !== null && Number.isFinite(total)) {
            return total.toLocaleString("zh-CN");
        }
        const keys = ["prompt_tokens", "completion_tokens", "input_tokens", "output_tokens"];
        const sum = keys.reduce((acc, key) => acc + (Number(usage[key]) || 0), 0);
        if (sum > 0) return sum.toLocaleString("zh-CN");
        return stringifyValue(usage);
    }

    function stringifyValue(value) {
        if (typeof value === "string") return value;
        try {
            return JSON.stringify(value);
        } catch (err) {
            return String(value);
        }
    }

    // ---------------------------------------------------------------- 事件绑定

    els.sendBtn.addEventListener("click", () => sendQuestion());

    els.input.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" || event.shiftKey) return;
        if (event.isComposing || event.keyCode === 229) return; // 中文输入法组词中
        event.preventDefault();
        sendQuestion();
    });

    els.input.addEventListener("input", autoGrow);

    document.querySelectorAll(".suggestion").forEach((button) => {
        button.addEventListener("click", () => {
            if (state.busy) return;
            els.input.value = button.dataset.q || button.textContent.trim();
            autoGrow();
            sendQuestion();
        });
    });

    els.messages.addEventListener("click", (event) => {
        const target = event.target.closest(".cite-ref");
        if (target) handleCiteRefClick(target.dataset.ref);
    });

    els.messages.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        const target = event.target.closest(".cite-ref");
        if (!target) return;
        event.preventDefault();
        handleCiteRefClick(target.dataset.ref);
    });

    els.modalClose.addEventListener("click", closeModal);
    els.modalOverlay.addEventListener("click", (event) => {
        if (event.target === els.modalOverlay) closeModal();
    });
    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && !els.modalOverlay.classList.contains("hidden")) {
            closeModal();
        }
    });

    autoGrow();
})();
