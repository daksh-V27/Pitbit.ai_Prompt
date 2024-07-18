import openai

def resume_to_json(resume):
    prompt = f"""
    You have to parse resumes into a structured JSON format. 
    Please take the resume provided below and extract key information such as personal details, education, work experience, skills, and certifications. 
    The output should be in JSON format always.

    Resume:
    {resume}

    Output JSON structure:
    {{
      "Personal_details": {{
        "Name": "",
        "Email": "",
        "Phone": "",
        "Address": ""
      }},
      "Education": [
        {{
          "Degree": "",
          "Institution": "",
          "Start_date": "",
          "End_date": ""
        }}
      ],
      "Work_experience": [
        {{
          "Wob_title": "",
          "Company": "",
          "Start_date": "",
          "End_date": "",
          "Responsibilities": [
            ""
          ]
        }}
      ],
      "Skills": [
        ""
      ],
      "Certifications": [
        {{
          "Name": "",
          "Issuing_organization": "",
          "Issue_date": ""
        }}
      ]
    }}
    """

    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=1500
    )

    return response.choices[0].text.strip()
