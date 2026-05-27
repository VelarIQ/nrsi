const DEFAULT_ALLOWED_DOMAINS = ["velariq.ai", "atheriongroup.com"];
const DEFAULT_SUPER_ADMIN_EMAILS = ["leighton@velariq.ai"];

export const ROLE_PRIORITY = {
  user: 10,
  admin: 50,
  super_admin: 100
};

export const PERMISSIONS_BY_ROLE = {
  user: ["dashboard:read"],
  admin: ["dashboard:read", "admin:read", "admin:ops"],
  super_admin: ["dashboard:read", "admin:read", "admin:ops", "admin:super"]
};

function parseCsvEnv(value, fallback) {
  const parsed = String(value || "")
    .split(",")
    .map((item) => item.trim().toLowerCase())
    .filter(Boolean);
  return parsed.length > 0 ? parsed : fallback;
}

export function getAllowedDomains() {
  return parseCsvEnv(process.env.AUTH_ALLOWED_DOMAINS, DEFAULT_ALLOWED_DOMAINS);
}

export function getSuperAdminEmails() {
  return parseCsvEnv(process.env.AUTH_SUPER_ADMIN_EMAILS, DEFAULT_SUPER_ADMIN_EMAILS);
}

export function getAdminEmails() {
  return parseCsvEnv(process.env.AUTH_ADMIN_EMAILS, []);
}

export function hasAllowedDomain(email = "") {
  const domain = String(email).toLowerCase().split("@")[1] || "";
  return getAllowedDomains().includes(domain);
}

export function roleForEmail(email = "") {
  const normalized = String(email).toLowerCase().trim();
  if (getSuperAdminEmails().includes(normalized)) return "super_admin";
  if (getAdminEmails().includes(normalized)) return "admin";
  return "user";
}

export function getPermissions(role = "user") {
  return PERMISSIONS_BY_ROLE[role] || PERMISSIONS_BY_ROLE.user;
}

export function hasRoleAtLeast(role = "user", minimumRole = "user") {
  return (ROLE_PRIORITY[role] || 0) >= (ROLE_PRIORITY[minimumRole] || 0);
}
