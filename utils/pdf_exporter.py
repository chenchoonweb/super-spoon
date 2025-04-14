from fpdf import FPDF

def export_answer_to_pdf(answer):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=answer)
    pdf_path = "/tmp/answer.pdf"
    pdf.output(pdf_path)
    return pdf_path
