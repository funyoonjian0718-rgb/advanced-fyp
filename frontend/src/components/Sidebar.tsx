import { NavLink, useNavigate } from "react-router-dom";
import { logout } from "../services/auth";

const navigation = [
    { to: "/dashboard", label: "Home", icon: "⌂", end: true },
    { to: "/chat", label: "Ask AI", icon: "✳", end: false },
    { to: "/history", label: "History", icon: "◷", end: false },
    { to: "/profile", label: "Profile", icon: "○", end: false },
];

function Sidebar() {
    const navigate = useNavigate();

    function handleLogout() {
        logout();
        navigate("/login", { replace: true });
    }

    return (
        <aside className="app-sidebar" aria-label="Main navigation">
            <a className="brand" href="/dashboard" onClick={(event) => { event.preventDefault(); navigate("/dashboard"); }}>
                <span className="brand-mark" aria-hidden="true">✳</span>
                <span>
                    <span className="brand-name">NutriGuide</span>
                    <span className="brand-subtitle" style={{ display: "block" }}>Your dietary companion</span>
                </span>
            </a>
            <div className="sidebar-label">Workspace</div>
            <nav className="sidebar-nav">
                {navigation.map((item) => (
                    <NavLink key={item.to} to={item.to} end={item.end} className={({ isActive }) => `sidebar-link${isActive ? " active" : ""}`}>
                        <span className="sidebar-icon" aria-hidden="true">{item.icon}</span>
                        <span>{item.label}</span>
                    </NavLink>
                ))}
            </nav>
            <div className="sidebar-bottom">
                <div className="sidebar-note">
                    <strong>Small steps matter 🌱</strong>
                    <p>Use evidence-based information to support your everyday food choices.</p>
                </div>
                <button type="button" onClick={handleLogout} className="sidebar-logout">↪ &nbsp; Sign out</button>
            </div>
        </aside>
    );
}

export default Sidebar;
