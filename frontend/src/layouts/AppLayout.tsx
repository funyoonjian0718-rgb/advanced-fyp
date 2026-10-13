import { Outlet } from "react-router-dom";
import Sidebar from "../components/Sidebar";
import Topbar from "../components/Topbar";

function AppLayout() {
    return (
        <div className="app-shell">
            <Sidebar />
            <div className="app-main-column">
                <Topbar />
                <main className="app-content">
                    <Outlet />
                </main>
            </div>
        </div>
    );
}

export default AppLayout;
