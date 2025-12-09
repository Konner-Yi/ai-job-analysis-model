import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

class ImprovedMLPClassifier(nn.Module):
    def __init__(self):
        super(ImprovedMLPClassifier, self).__init__()
        self.fc1 = nn.Linear(12, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 20)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def encode_feature(value, mapping, feature_name):
    if pd.isna(value) or value == '' or value is None:
        return -1
    try:
        idx = np.where(mapping == value)[0]
        return idx[0] if len(idx) > 0 else -1
    except:
        return -1

def encode_skill(value, encodings):
    if pd.isna(value) or value == '' or value is None:
        return -1
    for skill_col in ['Skill_1', 'Skill_2', 'Skill_3', 'Skill_4', 'Skill_5']:
        try:
            idx = np.where(encodings[skill_col] == value)[0]
            if len(idx) > 0:
                return idx[0]
        except:
            continue
    return -1

def encode_tool(value, encodings):
    if pd.isna(value) or value == '' or value is None:
        return -1
    for tool_col in ['Tools_1', 'Tools_2', 'Tools_3']:
        try:
            idx = np.where(encodings[tool_col] == value)[0]
            if len(idx) > 0:
                return idx[0]
        except:
            continue
    return -1

def predict_salary(job_data, model, encodings, fallback_mapping=None):
    model.eval()
    
    skills_list = [job_data.get(f'Skill_{i}') for i in range(1, 6)]
    tools_list = [job_data.get(f'Tools_{i}') for i in range(1, 4)]
    
    # Check if we have any skills/tools
    has_skills = any(skill for skill in skills_list if skill)
    has_tools = any(tool for tool in tools_list if tool)
    
    encoded_skills = []
    for skill in skills_list:
        if skill:
            encoded_skills.append(encode_skill(skill, encodings))
        else:
            encoded_skills.append(-1)
    
    encoded_tools = []
    for tool in tools_list:
        if tool:
            encoded_tools.append(encode_tool(tool, encodings))
        else:
            encoded_tools.append(-1)
    
    location_encoded = encode_feature(job_data.get('location'), encodings['location'], 'location')
    employment_encoded = encode_feature(job_data.get('employment_type'), encodings['employment_type'], 'employment_type')
    job_title_encoded = encode_feature(job_data.get('job_title'), encodings['job_title'], 'job_title')
    industry_encoded = encode_feature(job_data.get('industry'), encodings['industry'], 'industry')
    
    # Feature order must match: fskill_1-5, ftool_1-3, floc, ftype, ftitle, findustry
    # Note: fname (company_name) is NOT included in the model (last 12 columns exclude it)
    features = encoded_skills + encoded_tools + [
        location_encoded,
        employment_encoded,
        job_title_encoded,
        industry_encoded,
    ]
    
    input_tensor = torch.tensor([features], dtype=torch.float32)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # If no skills/tools provided and we have a fallback mapping, use it for better estimates
    if not has_skills and not has_tools and fallback_mapping is not None:
        job_title = job_data.get('job_title')
        industry = job_data.get('industry')
        fallback_key = (job_title, industry)
        
        if fallback_key in fallback_mapping:
            # Use fallback salary from training data
            fallback_avg = fallback_mapping[fallback_key]
            average_salary_usd = fallback_avg * 20000
            # Lower confidence since we're using fallback
            confidence = max(0.3, confidence * 0.7)
        else:
            # No fallback available, use model prediction
            average_salary_value = float(encodings['salary_categories'][predicted_class])
            average_salary_usd = average_salary_value * 20000
    else:
        # Skills/tools provided, use model prediction normally
        average_salary_value = float(encodings['salary_categories'][predicted_class])
        average_salary_usd = average_salary_value * 20000
    
    # Create a salary range around the average (typically ±$10,000)
    salary_range = (average_salary_usd - 10000, average_salary_usd + 10000)
    
    return {
        'predicted_category': predicted_class,
        'salary_range_usd': salary_range,
        'confidence': confidence
    }

def validate_input(value, valid_values, field_name):
    if not value or value.strip() == '':
        return None, None
    value = value.strip()
    if value not in valid_values:
        return None, f"'{value}' is not a valid {field_name}. This model only supports AI/ML job postings from the training data."
    return value, None

class SalaryPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Job Market Salary Predictor")
        self.root.geometry("600x700")
        self.root.resizable(False, False)
        
        self.model = None
        self.encodings = None
        self.load_model()
        
        self.create_widgets()
    
    def load_model(self):
        try:
            self.model = ImprovedMLPClassifier()
            self.model.load_state_dict(torch.load('model_state.pth'))
            self.model.eval()
            with open('encoding_mappings.pkl', 'rb') as f:
                self.encodings = pickle.load(f)
            self.default_location = self.encodings['location'][0] if len(self.encodings['location']) > 0 else None
            
            # Load fallback mapping if available
            try:
                with open('fallback_salary_mapping.pkl', 'rb') as f:
                    self.fallback_mapping = pickle.load(f)
            except FileNotFoundError:
                self.fallback_mapping = None
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.root.destroy()
    
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        title_label = ttk.Label(main_frame, text="AI Job Market Salary Predictor", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        subtitle_label = ttk.Label(main_frame, 
                                   text="Enter job details to predict salary range", 
                                   font=("Arial", 10))
        subtitle_label.grid(row=1, column=0, columnspan=2, pady=(0, 20))
        
        row = 2
        
        ttk.Label(main_frame, text="Job Title*:", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky=tk.W, pady=5)
        self.job_title_var = tk.StringVar()
        job_title_combo = ttk.Combobox(main_frame, textvariable=self.job_title_var, width=30)
        job_title_combo['values'] = sorted([str(x) for x in self.encodings['job_title'] if pd.notna(x)])
        job_title_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        ttk.Label(main_frame, text="Industry*:", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky=tk.W, pady=5)
        self.industry_var = tk.StringVar()
        industry_combo = ttk.Combobox(main_frame, textvariable=self.industry_var, width=30)
        industry_combo['values'] = sorted([str(x) for x in self.encodings['industry'] if pd.notna(x)])
        industry_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        ttk.Label(main_frame, text="Experience Level*:", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky=tk.W, pady=5)
        self.experience_level_var = tk.StringVar()
        experience_combo = ttk.Combobox(main_frame, textvariable=self.experience_level_var, width=30)
        experience_combo['values'] = ['Entry', 'Mid', 'Senior']
        experience_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        ttk.Label(main_frame, text="Company Size*:", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky=tk.W, pady=5)
        self.company_size_var = tk.StringVar()
        company_size_combo = ttk.Combobox(main_frame, textvariable=self.company_size_var, width=30)
        company_size_combo['values'] = ['Startup', 'Mid', 'Large']
        company_size_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        ttk.Label(main_frame, text="Employment Type*:", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky=tk.W, pady=5)
        self.employment_type_var = tk.StringVar()
        employment_combo = ttk.Combobox(main_frame, textvariable=self.employment_type_var, width=30)
        employment_combo['values'] = sorted([str(x) for x in self.encodings['employment_type'] if pd.notna(x)])
        employment_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        ttk.Label(main_frame, text="Skills (comma-separated):", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky=tk.W, pady=5)
        self.skills_var = tk.StringVar()
        skills_entry = ttk.Entry(main_frame, textvariable=self.skills_var, width=33)
        skills_entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        ttk.Label(main_frame, text="Tools (comma-separated):", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky=tk.W, pady=5)
        self.tools_var = tk.StringVar()
        tools_entry = ttk.Entry(main_frame, textvariable=self.tools_var, width=33)
        tools_entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        predict_btn = ttk.Button(main_frame, text="Predict Salary", command=self.predict)
        predict_btn.grid(row=row, column=0, columnspan=2, pady=20)
        row += 1
        
        ttk.Label(main_frame, text="Results:", font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
        row += 1
        
        self.result_text = scrolledtext.ScrolledText(main_frame, width=60, height=12, 
                                                     font=("Arial", 10), wrap=tk.WORD)
        self.result_text.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        info_label = ttk.Label(main_frame, 
                              text="* Required fields. This model only supports AI/ML job postings.", 
                              font=("Arial", 8), foreground="gray")
        info_label.grid(row=row, column=0, columnspan=2, pady=(10, 0))
        
        main_frame.columnconfigure(1, weight=1)
    
    def predict(self):
        self.result_text.delete(1.0, tk.END)
        
        job_title = self.job_title_var.get().strip()
        industry = self.industry_var.get().strip()
        experience_level = self.experience_level_var.get().strip()
        company_size = self.company_size_var.get().strip()
        employment_type = self.employment_type_var.get().strip()
        skills_input = self.skills_var.get().strip()
        tools_input = self.tools_var.get().strip()
        
        if not job_title or not industry or not experience_level or not company_size or not employment_type:
            messagebox.showwarning("Missing Information", 
                                 "Please fill in all required fields (Job Title, Industry, Experience Level, Company Size, Employment Type).")
            return
        
        job_title_val, error = validate_input(job_title, self.encodings['job_title'], "job title")
        if error:
            self.result_text.insert(tk.END, f"ERROR: {error}\n\n")
            self.result_text.insert(tk.END, f"Valid job titles include: Data Scientist, ML Engineer, NLP Engineer, Data Analyst, Computer Vision Engineer, Quant Researcher, AI Product Manager, AI Researcher\n")
            return
        
        industry_val, error = validate_input(industry, self.encodings['industry'], "industry")
        if error:
            self.result_text.insert(tk.END, f"ERROR: {error}\n\n")
            self.result_text.insert(tk.END, f"Valid industries include: Tech, Healthcare, Finance, E-commerce, Automotive, Retail, Education\n")
            return
        
        employment_type_val, error = validate_input(employment_type, self.encodings['employment_type'], "employment type")
        if error:
            self.result_text.insert(tk.END, f"ERROR: {error}\n\n")
            self.result_text.insert(tk.END, f"Valid employment types: Full-time, Contract, Remote, Internship\n")
            return
        
        skills_list = [s.strip() for s in skills_input.split(',') if s.strip()] if skills_input else []
        tools_list = [t.strip() for t in tools_input.split(',') if t.strip()] if tools_input else []
        
        # Validate skills and tools
        missing_skills = []
        missing_tools = []
        found_skills = []
        found_tools = []
        
        for skill in skills_list:
            encoded = encode_skill(skill, self.encodings)
            if encoded == -1:
                missing_skills.append(skill)
            else:
                found_skills.append(skill)
        
        for tool in tools_list:
            encoded = encode_tool(tool, self.encodings)
            if encoded == -1:
                missing_tools.append(tool)
            else:
                found_tools.append(tool)
        
        if missing_skills or missing_tools:
            warning = "WARNING: Some skills/tools were not found in training data:\n"
            if missing_skills:
                warning += f"  Skills not found: {', '.join(missing_skills)}\n"
            if missing_tools:
                warning += f"  Tools not found: {', '.join(missing_tools)}\n"
            warning += "  These will be ignored, which may affect prediction accuracy.\n\n"
            self.result_text.insert(tk.END, warning)
        
        # Warn if no skills/tools provided - model performs poorly without them
        if not skills_list and not tools_list:
            warning = "NOTE: No skills or tools provided. The model relies heavily on skills/tools for accurate predictions.\n"
            warning += "Predictions may be less accurate and similar across different job titles when skills/tools are missing.\n"
            warning += "For best results, please provide at least 1-2 relevant skills or tools.\n\n"
            self.result_text.insert(tk.END, warning)
        elif found_skills or found_tools:
            # Show which skills/tools were successfully found and will be used
            info = "Skills/tools being used in prediction:\n"
            if found_skills:
                info += f"  Skills: {', '.join(found_skills)}\n"
            if found_tools:
                info += f"  Tools: {', '.join(found_tools)}\n"
            info += "\n"
            self.result_text.insert(tk.END, info)
        
        job_data = {
            'Skill_1': skills_list[0] if len(skills_list) > 0 else None,
            'Skill_2': skills_list[1] if len(skills_list) > 1 else None,
            'Skill_3': skills_list[2] if len(skills_list) > 2 else None,
            'Skill_4': skills_list[3] if len(skills_list) > 3 else None,
            'Skill_5': skills_list[4] if len(skills_list) > 4 else None,
            'Tools_1': tools_list[0] if len(tools_list) > 0 else None,
            'Tools_2': tools_list[1] if len(tools_list) > 1 else None,
            'Tools_3': tools_list[2] if len(tools_list) > 2 else None,
            'location': self.default_location,
            'employment_type': employment_type_val,
            'job_title': job_title_val,
            'industry': industry_val,
        }
        
        try:
            result = predict_salary(job_data, self.model, self.encodings, self.fallback_mapping)
            
            self.result_text.insert(tk.END, "=" * 50 + "\n")
            self.result_text.insert(tk.END, "PREDICTION RESULTS\n")
            self.result_text.insert(tk.END, "=" * 50 + "\n\n")
            
            self.result_text.insert(tk.END, f"Job Title: {job_title}\n")
            self.result_text.insert(tk.END, f"Industry: {industry}\n")
            self.result_text.insert(tk.END, f"Experience Level: {experience_level}\n")
            self.result_text.insert(tk.END, f"Company Size: {company_size}\n")
            self.result_text.insert(tk.END, f"Employment Type: {employment_type}\n")
            if skills_list:
                self.result_text.insert(tk.END, f"Skills: {', '.join(skills_list)}\n")
            if tools_list:
                self.result_text.insert(tk.END, f"Tools: {', '.join(tools_list)}\n")
            
            self.result_text.insert(tk.END, "\n" + "-" * 50 + "\n\n")
            self.result_text.insert(tk.END, f"Predicted Salary Range:\n")
            self.result_text.insert(tk.END, f"${result['salary_range_usd'][0]:,.0f} - ${result['salary_range_usd'][1]:,.0f} USD\n\n")
            self.result_text.insert(tk.END, f"Confidence: {result['confidence']:.1%}\n")
            
        except Exception as e:
            self.result_text.insert(tk.END, f"ERROR: Failed to make prediction: {e}\n")
            self.result_text.insert(tk.END, "\nPlease ensure all inputs are valid AI/ML job postings from the training data.\n")

def main():
    root = tk.Tk()
    app = SalaryPredictorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
