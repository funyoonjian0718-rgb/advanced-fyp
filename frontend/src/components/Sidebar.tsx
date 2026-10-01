import { NavLink } from "react-router-dom";

function Sidebar() {
    return (
        <aside className="w-64 min-h-screen bg-gray-900 text-white p-5">
            <h1 className="text-xl font-bold mb-8">
                Advanced FYP
            </h1>

            <nav className="space-y-2">
                <NavLink
                    to="/dashboard"
                    className="block p-3 rounded hover:bg-gray-800"
                >
                    Dashboard
                </NavLink>

                <NavLink
                    to="/chat"
                    className="block p-3 rounded hover:bg-gray-800"
                >
                    Dietary Assistant
                </NavLink>

                <NavLink
                    to="/history"
                    className="block p-3 rounded hover:bg-gray-800"
                >
                    History
                </NavLink>

                <NavLink
                    to="/profile"
                    className="block p-3 rounded hover:bg-gray-800"
                >
                    Profile
                </NavLink>
            </nav>
        </aside>
    );
}

export default Sidebar;