import axios from "axios";

const api = axios.create({
    baseURL: "http://localhost:8000",
    headers: {
        "Content-Type": "application/json",
    },
});

// Axios interceptor
//
// This runs automatically before every request made using "api".
api.interceptors.request.use(
    (config) => {
        // Get the JWT that we saved after login.
        const token = localStorage.getItem("access_token");

        // If the user is logged in, attach the JWT to the request.
        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }

        // Return the modified request so Axios can send it.
        return config;
    },
    (error) => {
        // If Axios encounters an error while preparing the request,
        // pass that error to the caller.
        return Promise.reject(error);
    }
);

export default api;