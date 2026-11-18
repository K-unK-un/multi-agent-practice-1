/**
 * PolicyMind 前端交互：
 * - fetch + ReadableStream 手动解析 SSE（EventSource 不支持 POST）
 * - 流式渲染答案、引用高亮，并驱动右侧多 Agent 链路可视化
 */
(() => {
    "use strict";

    const SESSION_KEY = "policymind_session_id";
    const AUTH_KEY = "policymind_auth";
    const API_STREAM = "/api/chat/stream";

    const LEVEL_LABELS = {
        1: { text: "Lv.1", title: "一级：基础公开信息" },
        2: { text: "Lv.2", title: "二级：部门内信息" },
        3: { text: "Lv.3", title: "三级：中高层信息" },
        4: { text: "Lv.4", title: "四级：全部信息（同管理员查看权限）" },
    };

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
        authOverlay: $("authOverlay"),
        authForm: $("authForm"),
        authUsername: $("authUsername"),
        authPassword: $("authPassword"),
        authPassword2: $("authPassword2"),
        authPassword2Field: $("authPassword2Field"),
        authError: $("authError"),
        authHint: $("authHint"),
        authSubmit: $("authSubmit"),
        userArea: $("userArea"),
        userName: $("userName"),
        levelBadge: $("levelBadge"),
        adminBtn: $("adminBtn"),
        logoutBtn: $("logoutBtn"),
        adminOverlay: $("adminOverlay"),
        adminClose: $("adminClose"),
        adminRefresh: $("adminRefresh"),
        adminTbody: $("adminTbody"),
        adminError: $("adminError"),
        kbBtn: $("kbBtn"),
        kbOverlay: $("kbOverlay"),
        kbClose: $("kbClose"),
        kbRefresh: $("kbRefresh"),
        kbReindex: $("kbReindex"),
        kbUploadArea: $("kbUploadArea"),
        kbDropzone: $("kbDropzone"),
        kbFileInput: $("kbFileInput"),
        kbError: $("kbError"),
        kbTaskSection: $("kbTaskSection"),
        kbTaskList: $("kbTaskList"),
        kbTabDocs: $("kbTabDocs"),
        kbTabRecords: $("kbTabRecords"),
        kbDocsSection: $("kbDocsSection"),
        kbRecordsSection: $("kbRecordsSection"),
        kbTbody: $("kbTbody"),
        kbRecordsTbody: $("kbRecordsTbody"),
        kbDocStats: $("kbDocStats"),
        kbRecordStats: $("kbRecordStats"),
        kbTitle: $("kbTitle"),
        kbSub: $("kbSub"),
        kbHint: $("kbHint"),
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
        auth: loadAuth(),
        authMode: "login",
        adminUsers: [],
        kbDocs: [],
        kbTasks: [],
        kbRecords: [],
        kbTab: "docs",
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

    // ---------------------------------------------------------------- 认证

    function loadAuth() {
        try {
            const raw = localStorage.getItem(AUTH_KEY);
            if (!raw) return null;
            const parsed = JSON.parse(raw);
            if (parsed && parsed.token && parsed.user) return parsed;
        } catch (err) {
            /* ignore */
        }
        return null;
    }

    function saveAuth(auth) {
        state.auth = auth;
        try {
            if (auth) localStorage.setItem(AUTH_KEY, JSON.stringify(auth));
            else localStorage.removeItem(AUTH_KEY);
        } catch (err) {
            /* ignore */
        }
    }

    function authHeaders() {
        return state.auth && state.auth.token
            ? { Authorization: "Bearer " + state.auth.token }
            : {};
    }

    function showAuth(message) {
        els.authOverlay.classList.remove("hidden");
        if (message) showAuthError(message);
        els.authUsername.focus();
    }

    function hideAuth() {
        els.authOverlay.classList.add("hidden");
    }

    function showAuthError(text) {
        els.authError.textContent = text || "";
        els.authError.classList.toggle("hidden", !text);
    }

    function kbPermissions() {
        const user = state.auth && state.auth.user;
        const isAdmin = !!user && user.role === "admin";
        const level = user ? Number(user.level) || 1 : 1;
        // 入口对管理员 + 四级用户可见；只有管理员能上传/重建。
        return { view: isAdmin || level >= 4, manage: isAdmin };
    }

    function renderUserArea() {
        const auth = state.auth;
        if (!auth || !auth.user) {
            els.userName.textContent = "未登录";
            els.levelBadge.className = "level-badge";
            els.levelBadge.textContent = "";
            els.adminBtn.classList.add("hidden");
            els.kbBtn.classList.add("hidden");
            return;
        }
        const user = auth.user;
        els.userName.textContent = user.username || "用户";
        if (user.role === "admin") {
            els.userName.textContent += "（管理员）";
        }

        const level = Number(user.level) || 1;
        const info = LEVEL_LABELS[level] || LEVEL_LABELS[1];
        els.levelBadge.textContent = info.text;
        els.levelBadge.className = "level-badge level-" + level;
        els.levelBadge.title = info.title + (user.role === "admin" ? " · 管理员" : "");

        els.adminBtn.classList.toggle("hidden", user.role !== "admin");

        const perms = kbPermissions();
        els.kbBtn.classList.toggle("hidden", !perms.view);
        els.kbBtn.textContent = perms.manage ? "文档入库" : "知识库文档";
        if (!perms.view) els.kbOverlay.classList.add("hidden");
    }

    function logout() {
        saveAuth(null);
        state.kbTasks.forEach((task) => {
            if (task.timer) window.clearTimeout(task.timer);
        });
        state.kbTasks = [];
        els.kbOverlay.classList.add("hidden");
        els.adminOverlay.classList.add("hidden");
        hideAuth();
        showAuth();
        renderUserArea();
    }

    function handleAuthError(status, detail) {
        if (status === 401) {
            // token 失效/过期/被禁用：强制回到登录页
            logout();
            showAuthError(detail || "登录已失效，请重新登录");
        } else {
            showAuthError(detail || "操作失败（HTTP " + status + "）");
        }
    }

    async function submitAuth(event) {
        event.preventDefault();
        showAuthError("");

        const username = els.authUsername.value.trim();
        const password = els.authPassword.value;

        if (state.authMode === "register") {
            if (password !== els.authPassword2.value) {
                showAuthError("两次输入的密码不一致");
                return;
            }
            if (!/^[\w一-龥]{3,32}$/.test(username)) {
                showAuthError("用户名需为 3~32 位字母、数字、下划线或中文");
                return;
            }
        }

        const path = state.authMode === "register" ? "/api/auth/register" : "/api/auth/login";
        els.authSubmit.disabled = true;
        try {
            const response = await fetch(path, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ username: username, password: password }),
            });
            if (!response.ok) {
                let detail = "HTTP " + response.status;
                try {
                    const payload = await response.json();
                    if (payload && payload.detail) {
                        detail = typeof payload.detail === "string" ? payload.detail : stringifyValue(payload.detail);
                    }
                } catch (err) {
                    /* ignore */
                }
                showAuthError(detail);
                return;
            }
            const payload = await response.json();
            saveAuth({ token: payload.token, user: payload.user });
            hideAuth();
            renderUserArea();
            els.authForm.reset();
        } catch (err) {
            showAuthError("网络请求失败：" + (err && err.message ? err.message : err));
        } finally {
            els.authSubmit.disabled = false;
        }
    }

    // ---------------------------------------------------------------- 管理员面板

    function formatAdminDate(value) {
        if (value === null || value === undefined || value === "") return "—";
        return String(value).replace("T", " ").slice(0, 16);
    }

    async function loadAdminUsers() {
        els.adminError.classList.add("hidden");
        els.adminTbody.replaceChildren();
        const row = document.createElement("tr");
        row.innerHTML = '<td colspan="7" class="admin-empty">加载中…</td>';
        els.adminTbody.appendChild(row);
        try {
            const response = await fetch("/api/admin/users", { headers: authHeaders() });
            if (!response.ok) {
                let detail = "HTTP " + response.status;
                try {
                    const payload = await response.json();
                    if (payload && payload.detail) detail = payload.detail;
                } catch (err) { /* ignore */ }
                throw new Error(detail);
            }
            const payload = await response.json();
            state.adminUsers = Array.isArray(payload.users) ? payload.users : [];
            renderAdminTable();
        } catch (err) {
            els.adminTbody.replaceChildren();
            els.adminError.textContent = "加载失败：" + (err && err.message ? err.message : err);
            els.adminError.classList.remove("hidden");
        }
    }

    function renderAdminTable() {
        els.adminTbody.replaceChildren();
        if (!state.adminUsers.length) {
            const row = document.createElement("tr");
            row.innerHTML = '<td colspan="7" class="admin-empty">暂无用户</td>';
            els.adminTbody.appendChild(row);
            return;
        }
        state.adminUsers.forEach((user) => {
            const tr = document.createElement("tr");
            tr.dataset.uid = String(user.id);
            if (!user.is_active) tr.classList.add("row-disabled");

            const cells = [
                String(user.id),
                String(user.username),
                user.role === "admin" ? "管理员" : "普通用户",
                "", // 等级 select
                user.is_active ? "正常" : "已禁用",
                formatAdminDate(user.last_login_at),
                "", // 操作
            ];
            tr.append(...cells.map((text) => {
                const td = document.createElement("td");
                td.textContent = text;
                return td;
            }));

            // 等级选择器
            const levelTd = tr.children[3];
            levelTd.textContent = "";
            const select = document.createElement("select");
            select.className = "level-select";
            select.disabled = user.role === "admin";
            for (let level = 1; level <= 4; level++) {
                const option = document.createElement("option");
                option.value = String(level);
                option.textContent = "Lv." + level;
                if ((user.role === "admin" ? 4 : Number(user.level) || 1) === level) option.selected = true;
                select.appendChild(option);
            }
            if (user.role === "admin") {
                const hint = document.createElement("span");
                hint.className = "admin-fixed-hint";
                hint.textContent = " 管理员恒为 Lv.4";
                levelTd.append(select, hint);
            } else {
                levelTd.appendChild(select);
                select.addEventListener("change", () => updateUser(user, { level: Number(select.value) }));
            }

            // 操作按钮
            const opTd = tr.children[6];
            opTd.textContent = "";
            if (user.role !== "admin") {
                const toggle = document.createElement("button");
                toggle.type = "button";
                toggle.className = "admin-op";
                toggle.textContent = user.is_active ? "禁用" : "启用";
                toggle.addEventListener("click", () => updateUser(user, { is_active: !user.is_active }));

                const reset = document.createElement("button");
                reset.type = "button";
                reset.className = "admin-op";
                reset.textContent = "重置密码";
                reset.addEventListener("click", () => {
                    const pw = window.prompt("为「" + user.username + "」设置新密码（至少 6 位）：");
                    if (pw === null) return;
                    if (pw.length < 6) {
                        window.alert("密码至少 6 位");
                        return;
                    }
                    updateUser(user, { password: pw });
                });
                opTd.append(toggle, reset);
            } else {
                const tag = document.createElement("span");
                tag.className = "admin-self-tag";
                tag.textContent = "—";
                opTd.appendChild(tag);
            }

            els.adminTbody.appendChild(tr);
        });
    }

    async function updateUser(user, patch) {
        els.adminError.classList.add("hidden");
        const btnLabel = patch.level ? "Lv." + patch.level : (patch.is_active === false ? "禁用" : (patch.is_active === true ? "启用" : "重置密码"));
        try {
            const response = await fetch("/api/admin/users/" + encodeURIComponent(user.id), {
                method: "PATCH",
                headers: Object.assign({ "Content-Type": "application/json" }, authHeaders()),
                body: JSON.stringify(patch),
            });
            if (!response.ok) {
                let detail = "HTTP " + response.status;
                try {
                    const payload = await response.json();
                    if (payload && payload.detail) detail = payload.detail;
                } catch (err) { /* ignore */ }
                throw new Error(detail);
            }
            const payload = await response.json();
            const updated = payload.user;
            const index = state.adminUsers.findIndex((item) => String(item.id) === String(user.id));
            if (index !== -1 && updated) state.adminUsers[index] = updated;
            renderAdminTable();
            els.adminError.textContent = "已更新「" + user.username + "」：" + btnLabel;
            els.adminError.classList.add("admin-success");
            els.adminError.classList.remove("hidden");
        } catch (err) {
            els.adminError.textContent = "更新失败：" + (err && err.message ? err.message : err);
            els.adminError.classList.remove("admin-success");
            els.adminError.classList.remove("hidden");
        }
    }

    function openAdmin() {
        els.adminOverlay.classList.remove("hidden");
        loadAdminUsers();
    }

    // ---------------------------------------------------------------- 文档入库

    const KB_ACCEPTED = [".pdf", ".docx", ".md", ".txt"];
    const KB_ACTION_LABELS = {
        create: { text: "新建", tone: "create" },
        overwrite: { text: "覆盖更新", tone: "overwrite" },
        reingest_unchanged: { text: "重复入库", tone: "unchanged" },
        upgrade: { text: "版本升级", tone: "upgrade" },
        older_archived: { text: "历史版本", tone: "archived" },
        failed: { text: "失败", tone: "failed" },
    };

    function setKbNotice(text, success) {
        els.kbError.textContent = text || "";
        els.kbError.classList.toggle("hidden", !text);
        els.kbError.classList.toggle("admin-success", !!success);
    }

    function openKb() {
        const perms = kbPermissions();
        if (!perms.view) return;
        els.kbOverlay.classList.remove("hidden");
        els.kbUploadArea.classList.toggle("hidden", !perms.manage);
        els.kbReindex.classList.toggle("hidden", !perms.manage);
        els.kbTitle.textContent = perms.manage ? "文档入库" : "知识库文档";
        els.kbSub.textContent = perms.manage
            ? "上传制度文档后自动解析、切分并入库；文档列表仅管理员与四级用户可见。"
            : "您当前为四级用户，可查看已入库文档；上传与重建仅管理员可操作。";
        els.kbHint.textContent = perms.manage
            ? "上传后自动执行解析 → 清洗 → 条款切分 → 入库 → 向量与 BM25 索引；data/docs 中的文件可通过「重建索引」批量入库。"
            : "四级用户仅可查看文档列表，无法上传或重建索引。";
        setKbNotice("");
        renderKbTasks();
        switchKbTab(state.kbTab);
    }

    async function loadKbDocs() {
        els.kbTbody.replaceChildren();
        const loading = document.createElement("tr");
        loading.innerHTML = '<td colspan="7" class="admin-empty">加载中…</td>';
        els.kbTbody.appendChild(loading);
        try {
            const response = await fetch("/api/kb/docs?include_archived=true", { headers: authHeaders() });
            if (!response.ok) {
                let detail = "HTTP " + response.status;
                try {
                    const payload = await response.json();
                    if (payload && payload.detail) {
                        detail = typeof payload.detail === "string" ? payload.detail : stringifyValue(payload.detail);
                    }
                } catch (err) { /* ignore */ }
                if (response.status === 401) handleAuthError(401, detail);
                throw new Error(detail);
            }
            const payload = await response.json();
            state.kbDocs = Array.isArray(payload) ? payload : [];
            renderKbDocs();
        } catch (err) {
            els.kbTbody.replaceChildren();
            const failed = document.createElement("tr");
            failed.innerHTML = '<td colspan="7" class="admin-empty">加载失败</td>';
            els.kbTbody.appendChild(failed);
            els.kbDocStats.textContent = "";
            setKbNotice("加载文档列表失败：" + (err && err.message ? err.message : err));
        }
    }

    function renderKbDocs() {
        els.kbTbody.replaceChildren();
        const docs = state.kbDocs;
        const active = docs.filter((doc) => doc.status === "active").length;
        const clauses = docs.reduce((sum, doc) => sum + (Number(doc.clause_count) || 0), 0);
        els.kbDocStats.textContent = "共 " + docs.length + " 份 · 生效 " + active + " · 条款 " + clauses + " 条";

        if (!docs.length) {
            const empty = document.createElement("tr");
            empty.innerHTML = '<td colspan="7" class="admin-empty">暂无已入库文档，上传后会自动出现在这里</td>';
            els.kbTbody.appendChild(empty);
            return;
        }

        docs.forEach((doc) => {
            const tr = document.createElement("tr");
            if (doc.status !== "active") tr.classList.add("row-disabled");

            const titleTd = document.createElement("td");
            titleTd.className = "kb-title-cell";
            const title = document.createElement("span");
            title.className = "kb-doc-title";
            title.textContent = doc.title || "（未命名制度）";
            titleTd.appendChild(title);
            if (doc.source_name) {
                const file = document.createElement("span");
                file.className = "kb-doc-file";
                file.textContent = doc.source_name;
                file.title = doc.source_name;
                titleTd.appendChild(file);
            }
            tr.appendChild(titleTd);

            const versionTd = document.createElement("td");
            versionTd.textContent = doc.version || "—";
            tr.appendChild(versionTd);

            const dateTd = document.createElement("td");
            dateTd.textContent = doc.effective_date || "—";
            tr.appendChild(dateTd);

            const statusTd = document.createElement("td");
            const chip = document.createElement("span");
            chip.className = "kb-status " + (doc.status === "active" ? "active" : "archived");
            chip.textContent = doc.status === "active" ? "生效中" : "已归档";
            statusTd.appendChild(chip);
            tr.appendChild(statusTd);

            const scopeTd = document.createElement("td");
            scopeTd.textContent = Array.isArray(doc.scope) && doc.scope.length ? doc.scope.join("、") : "全员";
            tr.appendChild(scopeTd);

            const clauseTd = document.createElement("td");
            clauseTd.textContent = String(Number(doc.clause_count) || 0);
            tr.appendChild(clauseTd);

            const updatedTd = document.createElement("td");
            updatedTd.textContent = formatAdminDate(doc.updated_at || doc.created_at);
            tr.appendChild(updatedTd);

            els.kbTbody.appendChild(tr);
        });
    }

    function switchKbTab(tab) {
        state.kbTab = tab === "records" ? "records" : "docs";
        const docsActive = state.kbTab === "docs";
        els.kbTabDocs.classList.toggle("active", docsActive);
        els.kbTabRecords.classList.toggle("active", !docsActive);
        els.kbTabDocs.setAttribute("aria-selected", docsActive ? "true" : "false");
        els.kbTabRecords.setAttribute("aria-selected", docsActive ? "false" : "true");
        els.kbDocsSection.classList.toggle("hidden", !docsActive);
        els.kbRecordsSection.classList.toggle("hidden", docsActive);
        if (docsActive) loadKbDocs();
        else loadKbRecords();
    }

    function refreshKb() {
        if (state.kbTab === "records") loadKbRecords();
        else loadKbDocs();
    }

    async function loadKbRecords() {
        els.kbRecordsTbody.replaceChildren();
        const loading = document.createElement("tr");
        loading.innerHTML = '<td colspan="7" class="admin-empty">加载中…</td>';
        els.kbRecordsTbody.appendChild(loading);
        try {
            const response = await fetch("/api/kb/ingest-records?limit=100", { headers: authHeaders() });
            if (!response.ok) {
                let detail = "HTTP " + response.status;
                try {
                    const payload = await response.json();
                    if (payload && payload.detail) {
                        detail = typeof payload.detail === "string" ? payload.detail : stringifyValue(payload.detail);
                    }
                } catch (err) { /* ignore */ }
                if (response.status === 401) handleAuthError(401, detail);
                throw new Error(detail);
            }
            const payload = await response.json();
            state.kbRecords = Array.isArray(payload) ? payload : [];
            renderKbRecords();
        } catch (err) {
            els.kbRecordsTbody.replaceChildren();
            const failed = document.createElement("tr");
            failed.innerHTML = '<td colspan="7" class="admin-empty">加载失败</td>';
            els.kbRecordsTbody.appendChild(failed);
            els.kbRecordStats.textContent = "";
            setKbNotice("加载入库记录失败：" + (err && err.message ? err.message : err));
        }
    }

    function renderKbRecords() {
        els.kbRecordsTbody.replaceChildren();
        const records = state.kbRecords;
        els.kbRecordStats.textContent = records.length
            ? "最近 " + records.length + " 条"
            : "最近 100 条";

        if (!records.length) {
            const empty = document.createElement("tr");
            empty.innerHTML = '<td colspan="7" class="admin-empty">暂无入库记录</td>';
            els.kbRecordsTbody.appendChild(empty);
            return;
        }

        records.forEach((record) => {
            const tr = document.createElement("tr");
            if (record.ok === false) tr.classList.add("row-disabled");

            const timeTd = document.createElement("td");
            timeTd.className = "kb-record-time";
            timeTd.textContent = record.created_at || "—";
            tr.appendChild(timeTd);

            const docTd = document.createElement("td");
            const title = document.createElement("span");
            title.className = "kb-doc-title";
            title.textContent = record.title || "（未命名制度）";
            docTd.appendChild(title);
            const metaParts = [];
            if (record.version) metaParts.push(record.version);
            if (record.effective_date) metaParts.push(record.effective_date);
            if (record.document_status) {
                metaParts.push(record.document_status === "archived" ? "已归档" : "生效中");
            }
            if (metaParts.length) {
                const meta = document.createElement("span");
                meta.className = "kb-doc-file";
                meta.textContent = metaParts.join(" · ");
                docTd.appendChild(meta);
            }
            tr.appendChild(docTd);

            const idTd = document.createElement("td");
            idTd.className = "kb-record-id";
            if (record.doc_id) {
                idTd.textContent = String(record.doc_id).slice(0, 12) + "…";
                idTd.title = String(record.doc_id);
            } else {
                idTd.textContent = "—";
            }
            tr.appendChild(idTd);

            const sourceTd = document.createElement("td");
            sourceTd.className = "kb-record-source";
            sourceTd.textContent = record.source_name || "—";
            if (record.source_path) sourceTd.title = record.source_path;
            tr.appendChild(sourceTd);

            const operatorTd = document.createElement("td");
            operatorTd.textContent = record.operator_name || "—";
            tr.appendChild(operatorTd);

            const actionTd = document.createElement("td");
            const info = KB_ACTION_LABELS[record.action]
                || { text: record.action || "—", tone: "other" };
            const chip = document.createElement("span");
            chip.className = "kb-action kb-action-" + info.tone;
            chip.textContent = info.text;
            actionTd.appendChild(chip);
            tr.appendChild(actionTd);

            const noteTd = document.createElement("td");
            noteTd.className = "kb-record-note";
            const warnings = Array.isArray(record.warnings) ? record.warnings : [];
            noteTd.textContent = warnings.length
                ? warnings.join("；")
                : (record.ok === false ? "入库失败" : "—");
            if (noteTd.textContent.length > 120) {
                noteTd.title = noteTd.textContent;
                noteTd.textContent = noteTd.textContent.slice(0, 120) + "…";
            }
            tr.appendChild(noteTd);

            els.kbRecordsTbody.appendChild(tr);
        });
    }

    function renderKbTasks() {
        els.kbTaskSection.classList.toggle("hidden", !state.kbTasks.length);
        els.kbTaskList.replaceChildren();
        const statusLabels = { pending: "排队中", running: "处理中", done: "已完成", failed: "失败" };

        state.kbTasks.forEach((task) => {
            const card = document.createElement("div");
            card.className = "kb-task kb-task-" + (task.status || "pending");

            const head = document.createElement("div");
            head.className = "kb-task-head";

            const name = document.createElement("span");
            name.className = "kb-task-name";
            name.textContent = task.name || task.id;
            name.title = task.name || task.id;

            const badge = document.createElement("span");
            badge.className = "kb-task-status " + (task.status || "pending");
            badge.textContent = statusLabels[task.status] || task.status || "未知";

            head.append(name, badge);
            card.appendChild(head);

            const result = task.result;
            if (result && typeof result === "object") {
                const parts = [];
                if (result.documents !== undefined) parts.push("文档 " + result.documents);
                if (result.clauses !== undefined) parts.push("条款 " + result.clauses);
                if (result.rules !== undefined) parts.push("规则 " + result.rules);
                if (result.chroma_count !== undefined) parts.push("向量 " + result.chroma_count);
                if (parts.length) {
                    const summary = document.createElement("div");
                    summary.className = "kb-task-summary";
                    summary.textContent = parts.join(" · ");
                    card.appendChild(summary);
                }

                const warnings = Array.isArray(result.warnings) ? result.warnings : [];
                if (warnings.length) {
                    const warn = document.createElement("div");
                    warn.className = "kb-task-warn";
                    warn.textContent = "警告：" + warnings.slice(0, 3).join("；") + (warnings.length > 3 ? "；…" : "");
                    card.appendChild(warn);
                }
            }
            if (task.error) {
                const error = document.createElement("div");
                error.className = "kb-task-error";
                error.textContent = task.error;
                card.appendChild(error);
            }
            els.kbTaskList.appendChild(card);
        });
    }

    function trackKbTask(info) {
        const task = {
            id: info.id,
            name: info.name,
            kind: info.kind || "upload",
            status: "pending",
            result: null,
            error: null,
            pollFailures: 0,
            timer: null,
        };
        state.kbTasks.unshift(task);
        renderKbTasks();
        pollKbTask(task);
        return task;
    }

    async function pollKbTask(task) {
        if (!task || task.status === "done" || task.status === "failed") return;
        if (task.timer) {
            window.clearTimeout(task.timer);
            task.timer = null;
        }
        try {
            const response = await fetch("/api/kb/tasks/" + encodeURIComponent(task.id), { headers: authHeaders() });
            if (response.status === 401) {
                handleAuthError(401, "登录已失效，请重新登录");
                return;
            }
            if (response.status === 403) {
                task.status = "failed";
                task.error = "权限不足：仅管理员可查看入库任务";
                renderKbTasks();
                return;
            }
            if (!response.ok) {
                let detail = "HTTP " + response.status;
                try {
                    const payload = await response.json();
                    if (payload && payload.detail) detail = payload.detail;
                } catch (err) { /* ignore */ }
                if (response.status === 404) {
                    task.status = "failed";
                    task.error = "任务不存在（服务可能已重启）";
                    renderKbTasks();
                    return;
                }
                throw new Error(detail);
            }
            const payload = await response.json();
            task.status = String(payload.status || task.status);
            task.result = payload.result || null;
            task.error = payload.error || null;
            task.pollFailures = 0;
            renderKbTasks();

            if (task.status === "done" || task.status === "failed") {
                if (!els.kbOverlay.classList.contains("hidden")) refreshKb();
                return;
            }
        } catch (err) {
            task.pollFailures = (task.pollFailures || 0) + 1;
            if (task.pollFailures >= 20) {
                task.status = "failed";
                task.error = "无法获取任务状态：" + (err && err.message ? err.message : err);
                renderKbTasks();
                return;
            }
        }
        task.timer = window.setTimeout(() => pollKbTask(task), 1500);
    }

    async function uploadKbFile(file) {
        const form = new FormData();
        form.append("file", file, file.name);
        const response = await fetch("/api/kb/upload", {
            method: "POST",
            headers: authHeaders(),
            body: form,
        });
        if (!response.ok) {
            let detail = "HTTP " + response.status;
            try {
                const payload = await response.json();
                if (payload && payload.detail) {
                    detail = typeof payload.detail === "string" ? payload.detail : stringifyValue(payload.detail);
                }
            } catch (err) { /* ignore */ }
            if (response.status === 401) handleAuthError(401, detail);
            throw new Error(detail);
        }
        const payload = await response.json();
        if (!payload || !payload.task_id) throw new Error("服务端未返回 task_id");
        trackKbTask({ id: payload.task_id, name: file.name, kind: "upload" });
        setKbNotice("已上传「" + file.name + "」，正在自动入库…", true);
    }

    async function uploadKbFiles(fileList) {
        const perms = kbPermissions();
        if (!perms.manage) {
            setKbNotice("仅管理员可上传文档");
            return;
        }
        const files = Array.from(fileList || []);
        if (!files.length) return;
        setKbNotice("");
        for (const file of files) {
            const lower = String(file.name || "").toLowerCase();
            if (!KB_ACCEPTED.some((suffix) => lower.endsWith(suffix))) {
                setKbNotice("已跳过不支持的文件：" + file.name);
                continue;
            }
            try {
                await uploadKbFile(file);
            } catch (err) {
                setKbNotice("上传「" + file.name + "」失败：" + (err && err.message ? err.message : err));
            }
        }
        els.kbFileInput.value = "";
    }

    async function reindexKb() {
        const perms = kbPermissions();
        if (!perms.manage) return;
        if (!window.confirm("重建索引会重新扫描 data/docs 与已上传文件并重新生成向量索引，可能耗时较长，确定继续？")) {
            return;
        }
        setKbNotice("");
        try {
            const response = await fetch("/api/kb/reindex", { method: "POST", headers: authHeaders() });
            if (!response.ok) {
                let detail = "HTTP " + response.status;
                try {
                    const payload = await response.json();
                    if (payload && payload.detail) detail = payload.detail;
                } catch (err) { /* ignore */ }
                if (response.status === 401) handleAuthError(401, detail);
                throw new Error(detail);
            }
            const payload = await response.json();
            if (!payload || !payload.task_id) throw new Error("服务端未返回 task_id");
            trackKbTask({ id: payload.task_id, name: "全量重建索引", kind: "reindex" });
            setKbNotice("已提交重建索引任务，正在后台处理…", true);
        } catch (err) {
            setKbNotice("提交重建索引失败：" + (err && err.message ? err.message : err));
        }
    }

    function bindKbUpload() {
        els.kbFileInput.addEventListener("change", () => uploadKbFiles(els.kbFileInput.files));
        ["dragenter", "dragover"].forEach((name) => {
            els.kbDropzone.addEventListener(name, (event) => {
                event.preventDefault();
                els.kbDropzone.classList.add("dragging");
            });
        });
        ["dragleave", "dragend", "drop"].forEach((name) => {
            els.kbDropzone.addEventListener(name, (event) => {
                event.preventDefault();
                els.kbDropzone.classList.remove("dragging");
            });
        });
        els.kbDropzone.addEventListener("drop", (event) => {
            const files = event.dataTransfer && event.dataTransfer.files;
            if (files && files.length) uploadKbFiles(files);
        });
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
            headers: Object.assign({
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            }, authHeaders()),
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
            if (response.status === 401) {
                logout();
                showAuthError(detail || "登录已失效，请重新登录");
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

    // 认证：登录/注册切换
    document.querySelectorAll(".auth-tab").forEach((tab) => {
        tab.addEventListener("click", () => {
            state.authMode = tab.dataset.mode;
            document.querySelectorAll(".auth-tab").forEach((item) => {
                item.classList.toggle("active", item === tab);
            });
            els.authSubmit.textContent = state.authMode === "register" ? "注 册" : "登 录";
            els.authPassword2Field.classList.toggle("hidden", state.authMode !== "register");
            els.authPassword2.required = state.authMode === "register";
            els.authPassword.autocomplete = state.authMode === "register" ? "new-password" : "current-password";
            els.authHint.classList.toggle("hidden", state.authMode === "register");
            showAuthError("");
        });
    });

    els.authForm.addEventListener("submit", submitAuth);
    els.logoutBtn.addEventListener("click", logout);

    // 管理员面板
    els.adminBtn.addEventListener("click", openAdmin);
    els.adminClose.addEventListener("click", () => els.adminOverlay.classList.add("hidden"));
    els.adminOverlay.addEventListener("click", (event) => {
        if (event.target === els.adminOverlay) els.adminOverlay.classList.add("hidden");
    });
    els.adminRefresh.addEventListener("click", loadAdminUsers);

    // 文档入库面板
    els.kbBtn.addEventListener("click", openKb);
    els.kbClose.addEventListener("click", () => els.kbOverlay.classList.add("hidden"));
    els.kbOverlay.addEventListener("click", (event) => {
        if (event.target === els.kbOverlay) els.kbOverlay.classList.add("hidden");
    });
    els.kbRefresh.addEventListener("click", refreshKb);
    els.kbReindex.addEventListener("click", reindexKb);
    els.kbTabDocs.addEventListener("click", () => switchKbTab("docs"));
    els.kbTabRecords.addEventListener("click", () => switchKbTab("records"));
    bindKbUpload();

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
        if (event.key !== "Escape") return;
        if (!els.modalOverlay.classList.contains("hidden")) closeModal();
        if (!els.adminOverlay.classList.contains("hidden")) els.adminOverlay.classList.add("hidden");
        if (!els.kbOverlay.classList.contains("hidden")) els.kbOverlay.classList.add("hidden");
    });

    // ---------------------------------------------------------------- 启动

    async function boot() {
        renderUserArea();
        if (!state.auth) {
            showAuth();
            return;
        }
        // 静默校验 token 并刷新用户信息（管理员改等级后重新进入页面即生效）
        try {
            const response = await fetch("/api/auth/me", { headers: authHeaders() });
            if (response.ok) {
                const user = await response.json();
                saveAuth({ token: state.auth.token, user: user });
                renderUserArea();
            } else if (response.status === 401) {
                logout();
                showAuth();
            }
        } catch (err) {
            /* 网络异常不强制登出 */
        }
    }

    autoGrow();
    boot();
})();
