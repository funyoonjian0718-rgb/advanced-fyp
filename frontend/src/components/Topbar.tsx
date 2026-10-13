import { useLocation } from "react-router-dom";

const titles: Record<string, { title: string; kicker: string }> = {
    "/dashboard": { title: "Your overview", kicker: "Workspace" },
    "/chat": { title: "Dietary assistant", kicker: "Ask with confidence" },
    "/history": { title: "Your conversations", kicker: "Your activity" },
    "/profile": { title: "Your profile", kicker: "Personal preferences" },
};

function Topbar() {
    const location = useLocation();
    const page = titles[location.pathname] ?? titles["/dashboard"];
    let email = "Your account";
    try {
        const storedUser = localStorage.getItem("user");
        if (storedUser) {
            const parsed: unknown = JSON.parse(storedUser);
            if (typeof parsed === "object" && parsed !== null && "email" in parsed && typeof parsed.email === "string") email = parsed.email;
        }
    } catch { /* Ignore invalid cached user data; the page remains usable. */ }

    return (
        <header className="app-topbar">
            <div>
                <div className="topbar-kicker">{page.kicker}</div>
                <h2 className="topbar-title">{page.title}</h2>
            </div>
            <div className="topbar-user">
                <span className="topbar-avatar" aria-hidden="true">{email.charAt(0).toUpperCase()}</span>
                <span>{email}</span>
            </div>
        </header>
    );
}

export default Topbar;
