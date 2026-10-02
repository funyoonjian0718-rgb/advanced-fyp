import api from "./api";

export interface UserProfile {
    id: number;
    user_id: number;
    full_name: string | null;
    age: number | null;
    gender: string | null;
    height_cm: number | null;
    weight_kg: number | null;
    activity_level: string | null;
    dietary_preference: string | null;
    allergies: string | null;
    updated_at: string;
}

export type UserProfileUpdate = Omit<
    UserProfile,
    "id" | "user_id" | "updated_at"
>;

export async function getProfile(): Promise<UserProfile> {
    const response = await api.get<UserProfile>("/api/profile");
    return response.data;
}

export async function updateProfile(
    data: UserProfileUpdate
): Promise<UserProfile> {
    const response = await api.put<UserProfile>("/api/profile", data);
    return response.data;
}
