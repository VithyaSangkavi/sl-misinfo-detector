import axios from "axios";

const API_BASE_URL = "http://127.0.0.1:8000";

export async function predictHeadline(headline) {
  const response = await axios.post(`${API_BASE_URL}/predict`, {
    headlines: [headline],
  });
  return response.data[0]; // one item
}
