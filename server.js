const express = require('express');
const cors = require('cors');
const ort = require('onnxruntime-node');
const { OpenAI } = require('openai');
const dotenv = require('dotenv');
const { createClient } = require('@supabase/supabase-js');  
const app = express();
//update
app.use(cors());

app.use(express.json());

dotenv.config();

const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,  
});

async function detectPhishing(urls) {
    try {
        const model_path = "./model.onnx";
        const session = await ort.InferenceSession.create(model_path);
        const tensor = new ort.Tensor('string', urls, [urls.length]);
        const results = await session.run({ "inputs": tensor });
        const probas = results['probabilities'].data;

        return urls.map((url, index) => {
            const proba = probas[index * 2 + 1];
            return {
                url,
                probability: proba * 100,  
            };
        });
    } catch (e) {
        console.error("Failed to infer ONNX model:", e);
        throw new Error(`Model inference failed: ${e.message}`);
    }
}

// Add data to Supabase
async function addDataToSupabase(url, isPhishing, probability) {
    try {
        console.log('Adding data to Supabase:', { url, isPhishing, probability });
        
        const { data, error } = await supabase
            .from('phishing_data')
            .insert([{ url, is_phishing: isPhishing, probability }]);

        if (error) {
            throw error;
        }

        console.log('Data added to Supabase successfully:', data);
    } catch (error) {
        console.error('Error adding data to Supabase:', error);
        throw new Error(`Error adding data to Supabase: ${error.message}`);
    }
}

// Fetch latest data from Supabase
async function fetchLatestData() {
    try {
        const { data, error } = await supabase
            .from('phishing_data')
            .select('*')
            .order('timestamp', { ascending: false })
            .limit(1);

        if (error) {
            throw error;
        }

        return data.length > 0 ? data[0] : null;
    } catch (error) {
        console.error('Error fetching latest data from Supabase:', error);
        throw new Error(`Error fetching data: ${error.message}`);
    }
}

// Handle POST requests to /detect-phishing
app.post('/detect-phishing', async (req, res) => {
    const { url } = req.body;

    if (!url) {
        return res.status(400).json({ error: 'URL is required' });
    }

    try {
        const results = await detectPhishing([url]);

        const isPhishing = results[0].probability > 50;

        await addDataToSupabase(url, isPhishing, results[0].probability);

        return res.json({
            isPhishing,
            probability: results[0].probability,
        });
    } catch (e) {
        console.error("Error in /detect-phishing:", e);
        return res.status(500).json({ error: 'Failed to process the request', message: e.message });
    }
});

// Handle POST requests to /gpt-response
app.post('/gpt-response', async (req, res) => {
    const { prompt } = req.body;

    if (!prompt) {
        return res.status(400).json({ error: 'Prompt is required' });
    }

    try {
        const response = await openai.chat.completions.create({
            model: 'gpt-3.5-turbo',
            messages: [{ role: 'user', content: prompt }],
        });

        return res.json({
            reply: response.choices[0].message.content,
        });
    } catch (e) {
        console.error("Error calling GPT model:", e);
        return res.status(500).json({ error: 'Failed to get response from GPT model', message: e.message });
    }
});

// Handle POST requests to /add-data
app.post('/add-data', async (req, res) => {
    const { url, probability, isPhishing } = req.body;

    if (!url || probability === undefined || isPhishing === undefined) {
        return res.status(400).json({ error: 'URL, probability, and isPhishing are required' });
    }

    try {
        await addDataToSupabase(url, isPhishing, probability); 
        return res.json({ message: 'Data added successfully' });
    } catch (e) {
        console.error("Error in /add-data:", e);
        return res.status(500).json({ error: 'Failed to add data', message: e.message });
    }
});


// Endpoint to fetch the latest phishing data
app.get('/fetch-latest', async (req, res) => {
    try {
        const latestData = await fetchLatestData();
        if (latestData) {
            res.json(latestData);
        } else {
            res.status(404).json({ error: 'No data found' });
        }
    } catch (e) {
        console.error("Error in /fetch-latest:", e);
        res.status(500).json({ error: 'Failed to fetch latest data', message: e.message });
    }
});

const port = process.env.PORT || 5000;
app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});
