const menuButton = document.getElementById("menuButton");
const menu = document.getElementById("menu");
const SESSION_KEY = "nrsi_site_session_v1";

if (menuButton && menu) {
  menuButton.addEventListener("click", () => {
    menu.classList.toggle("open");
  });
}

const yearNode = document.getElementById("year");
if (yearNode) {
  yearNode.textContent = String(new Date().getFullYear());
}

function getSession() {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function setSession(session) {
  localStorage.setItem(SESSION_KEY, JSON.stringify(session));
}

function clearSession() {
  localStorage.removeItem(SESSION_KEY);
}

function redirectToLogin(nextPage) {
  const target = encodeURIComponent(nextPage || "dashboard.html");
  window.location.href = `./login.html?next=${target}`;
}

function hydrateAuthUi(session) {
  const loginLink = document.getElementById("navLoginLink");
  const dashLink = document.getElementById("navDashboardLink");
  const adminLink = document.getElementById("navAdminLink");
  const logoutLinks = document.querySelectorAll(".js-logout");
  const emailNodes = document.querySelectorAll("[data-auth-email]");
  const roleNodes = document.querySelectorAll("[data-auth-role]");
  const timeNodes = document.querySelectorAll("[data-auth-time]");

  const hasSession = Boolean(session);
  if (loginLink) loginLink.style.display = hasSession ? "none" : "inline-block";
  if (dashLink) dashLink.style.display = hasSession ? "inline-block" : "none";
  if (adminLink) adminLink.style.display = hasSession && session.role === "admin" ? "inline-block" : "none";
  logoutLinks.forEach((el) => {
    el.style.display = hasSession ? "inline-block" : "none";
    el.addEventListener("click", (event) => {
      event.preventDefault();
      clearSession();
      window.location.href = "./login.html";
    });
  });

  emailNodes.forEach((node) => {
    if (session?.email) node.textContent = session.email;
  });
  roleNodes.forEach((node) => {
    if (session?.role) node.textContent = session.role;
  });
  timeNodes.forEach((node) => {
    if (session?.loginAt) node.textContent = new Date(session.loginAt).toLocaleString();
  });
}

function enforceAuth(session) {
  const body = document.body;
  const requireAuth = body?.dataset?.requireAuth === "true";
  const requireRole = body?.dataset?.requireRole;
  const currentPath = window.location.pathname.split("/").pop() || "index.html";

  if (requireAuth && !session) {
    redirectToLogin(currentPath);
    return;
  }
  if (requireRole && session?.role !== requireRole) {
    window.location.href = "./dashboard.html";
  }
}

function bindLoginForm() {
  const form = document.getElementById("loginForm");
  if (!form) return;

  const emailInput = document.getElementById("loginEmail");
  const passInput = document.getElementById("loginPassword");
  const roleInput = document.getElementById("loginRole");
  const nextInput = document.getElementById("loginNext");
  const params = new URLSearchParams(window.location.search);
  const requestedNext = params.get("next");
  if (requestedNext && nextInput) {
    nextInput.value = requestedNext;
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const email = emailInput?.value?.trim() || "";
    const password = passInput?.value || "";
    let role = roleInput?.value || "user";

    if (!email || password.length < 6) {
      alert("Please enter a valid email and password.");
      return;
    }

    if (email.startsWith("admin@") || email.includes("+admin")) {
      role = "admin";
    }

    const session = {
      email,
      role,
      loginAt: new Date().toISOString()
    };
    setSession(session);

    const nextPath = (nextInput?.value || "dashboard.html").replace("./", "");
    if (nextPath === "admin.html" && role !== "admin") {
      window.location.href = "./dashboard.html";
      return;
    }
    window.location.href = `./${nextPath}`;
  });
}

const session = getSession();
hydrateAuthUi(session);
enforceAuth(session);
bindLoginForm();
