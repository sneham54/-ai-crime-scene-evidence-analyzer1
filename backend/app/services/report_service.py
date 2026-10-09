"""
Professional Forensic PDF Report Generator
Uses ReportLab – generates multi-section investigative reports.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable, KeepTogether
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
import os
import uuid
from xml.sax.saxutils import escape
from flask import current_app


class ReportService:
    def generate(
        self,
        case_data: dict,
        detections: list,
        risk: dict,
        custody_logs: list = None,
        annotated_image_path: str = None,
        investigator: str = "investigator",
        video_result: dict = None,
        source_filename: str = None,
        analyzed_at: str = None,
    ):
        """
        Generate a professional multi-section PDF forensic report.
        Returns (absolute_path, filename)
        """
        custody_logs = custody_logs or []

        reports_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], "reports")
        os.makedirs(reports_dir, exist_ok=True)

        report_name = f"forensic_report_{uuid.uuid4().hex[:10]}.pdf"
        report_path = os.path.join(reports_dir, report_name)

        doc = SimpleDocTemplate(
            report_path,
            pagesize=A4,
            rightMargin=1.6 * cm,
            leftMargin=1.6 * cm,
            topMargin=1.4 * cm,
            bottomMargin=1.4 * cm,
            title="Forensic Analysis Report",
            author="AI Crime Scene Evidence Analyzer",
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Heading1"],
            fontSize=16,
            alignment=TA_CENTER,
            spaceAfter=4,
            textColor=colors.HexColor("#0d1b2a"),
            fontName="Helvetica-Bold",
        )
        subtitle_style = ParagraphStyle(
            "Subtitle",
            parent=styles["Normal"],
            fontSize=12,
            alignment=TA_CENTER,
            spaceAfter=10,
            textColor=colors.HexColor("#1b263b"),
            fontName="Helvetica-Bold",
        )
        heading_style = ParagraphStyle(
            "SectionHead",
            parent=styles["Heading2"],
            fontSize=11,
            spaceBefore=12,
            spaceAfter=6,
            textColor=colors.HexColor("#1b263b"),
            fontName="Helvetica-Bold",
            borderPadding=3,
        )
        normal = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            alignment=TA_JUSTIFY,
        )
        small = ParagraphStyle(
            "Small",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.HexColor("#415a77"),
        )
        footer_style = ParagraphStyle(
            "FooterNote",
            parent=styles["Normal"],
            fontSize=7.5,
            textColor=colors.grey,
            alignment=TA_CENTER,
            spaceBefore=16,
        )

        story = []

        # ── Header ──────────────────────────────────────────────
        story.append(Paragraph("AI CRIME SCENE EVIDENCE ANALYZER", title_style))
        story.append(Paragraph("FORENSIC ANALYSIS REPORT", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1b263b"), spaceAfter=8))

        meta_data = [
            ["Report ID", report_name],
            ["Generated (UTC)", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")],
            ["Case ID", str(case_data.get("case_id", "N/A"))],
            ["Investigator", investigator],
            ["Case Description", escape(str(case_data.get("description", "Automated crime-scene evidence analysis")))],
            ["File Name", escape(os.path.basename(source_filename)) if source_filename else "Not provided"],
            ["Analysis Date/Time", escape(str(analyzed_at)) if analyzed_at else "Not provided"],
        ]
        meta_table = Table(meta_data, colWidths=[2.2 * inch, 4.5 * inch])
        meta_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1b263b")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 8))

        # ── 1. Executive Summary / Risk ─────────────────────────
        story.append(Paragraph("1. RISK ANALYSIS SUMMARY", heading_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#778da9"), spaceAfter=6))

        overall = risk.get("overall_risk", 0)
        label = risk.get("risk_label", "Unknown")
        threat = risk.get("threat_level", "N/A")
        severity = risk.get("severity", "N/A")

        risk_rows = [
            ["Metric", "Value"],
            ["Overall Risk Score", f"{overall} / 100"],
            ["Risk Label", str(label)],
            ["Threat Level", str(threat)],
            ["Severity", str(severity)],
            ["Probability", str(risk.get("probability", "N/A"))],
            ["Weapons Detected", str(risk.get("weapon_count", 0))],
            ["Persons Detected", str(risk.get("person_count", risk.get("persons", "—")))],
        ]
        risk_table = Table(risk_rows, colWidths=[2.8 * inch, 3.9 * inch])
        risk_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b263b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#e0e7ff")),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94a3b8")),
            ("PADDING", (0, 0), (-1, -1), 5),
            ("ALIGN", (1, 1), (1, -1), "CENTER"),
        ]))
        story.append(risk_table)

        reasons = risk.get("reasons") or []
        if reasons:
            story.append(Spacer(1, 4))
            story.append(Paragraph("<b>Key risk factors:</b>", normal))
            for r in reasons:
                story.append(Paragraph(f"• {r}", normal))

        # ── 2. Detected Objects ─────────────────────────────────
        story.append(Paragraph("2. DETECTED EVIDENCE OBJECTS", heading_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#778da9"), spaceAfter=6))

        if detections:
            det_header = ["#", "Class", "Category", "YOLO Confidence", "Bounding Box (x1,y1,x2,y2)"]
            det_data = [det_header]
            for i, d in enumerate(detections, 1):
                bbox = d.get("bbox", [])
                bbox_str = ", ".join(f"{x:.0f}" if isinstance(x, float) else str(x) for x in bbox)
                det_data.append([
                    str(i),
                    str(d.get("class", "")),
                    str(d.get("category", "")),
                    f"{float(d.get('confidence', 0)):.2f}",
                    bbox_str,
                ])
            det_table = Table(det_data, colWidths=[0.4*inch, 1.3*inch, 1.3*inch, 0.9*inch, 2.8*inch])
            det_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1565c0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94a3b8")),
                ("PADDING", (0, 0), (-1, -1), 4),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("ALIGN", (3, 1), (3, -1), "CENTER"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
            ]))
            story.append(det_table)

            # Category summary
            cats = {}
            for d in detections:
                c = d.get("category", "Others")
                cats[c] = cats.get(c, 0) + 1
            summary_line = "  |  ".join(f"{k}: {v}" for k, v in sorted(cats.items()))
            story.append(Spacer(1, 4))
            story.append(Paragraph(f"<b>Category summary:</b> {summary_line}", small))
            story.append(Paragraph(
                "YOLO confidence is the model's confidence for each individual detection; "
                "it is not overall model accuracy. Accuracy requires evaluation against a labeled test set.",
                small,
            ))
        else:
            story.append(Paragraph("No objects were detected in the submitted media.", normal))

        if video_result:
            classification = video_result.get("classification") or video_result.get("event_classification") or {}
            story.append(Paragraph("VIDEO EVENT CLASSIFICATION", heading_style))
            event_label = escape(str(classification.get("label", "Unavailable")))
            confidence = classification.get("confidence")
            confidence_text = f"{float(confidence):.1%}" if confidence is not None else "Unavailable"
            story.append(Paragraph(f"Video classifier result: <b>{event_label}</b> · Confidence: {confidence_text}", normal))
            story.append(Paragraph("This event classification is produced by the whole-video classifier. YOLO object detections are separate visual evidence and do not establish that the event occurred.", normal))
            timeline = video_result.get("timeline") or video_result.get("event_timeline") or []
            story.append(Paragraph("EVIDENCE TIMELINE", heading_style))
            if timeline:
                timeline_rows = [["Timestamp", "Frame", "Detection", "YOLO confidence"]]
                for item in timeline:
                    seconds = item.get("timestamp_sec")
                    timestamp = f"{float(seconds):.1f} s" if seconds is not None else "Unavailable"
                    timeline_rows.append([timestamp, str(item.get("frame_number", "")), escape(str(item.get("class_name", item.get("class", "")))), f"{float(item.get('confidence', 0)):.1%}"])
                timeline_table = Table(timeline_rows, colWidths=[1.1*inch, 0.8*inch, 2.8*inch, 1.4*inch], repeatRows=1)
                timeline_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b263b")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94a3b8")),
                    ("PADDING", (0, 0), (-1, -1), 4),
                ]))
                story.append(timeline_table)
            else:
                story.append(Paragraph("No YOLO detections were recorded in the sampled frames.", normal))
            summary = video_result.get("investigation_summary")
            if summary:
                story.append(Paragraph("INVESTIGATION SUMMARY", heading_style))
                story.append(Paragraph(escape(str(summary)), normal))

        # ── 3. Annotated Image ──────────────────────────────────
        story.append(Paragraph("3. ANNOTATED EVIDENCE IMAGE", heading_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#778da9"), spaceAfter=6))

        img_embedded = False
        if annotated_image_path:
            # Handle both absolute and relative paths, Windows/Linux
            candidates = [
                annotated_image_path,
                os.path.normpath(annotated_image_path),
            ]
            # If only filename given, look under uploads/annotated
            if not os.path.isabs(annotated_image_path):
                candidates.append(
                    os.path.join(current_app.config["UPLOAD_FOLDER"], "annotated", os.path.basename(annotated_image_path))
                )
                candidates.append(os.path.join(current_app.config["UPLOAD_FOLDER"], annotated_image_path))

            for path in candidates:
                if path and os.path.isfile(path):
                    try:
                        img = Image(path, width=5.8 * inch, height=3.6 * inch, kind="proportional")
                        story.append(img)
                        story.append(Paragraph(f"<i>Source: {os.path.basename(path)}</i>", small))
                        img_embedded = True
                        break
                    except Exception as e:
                        print(f"Could not embed image {path}: {e}")
        if not img_embedded:
            story.append(Paragraph(
                "Annotated image was not available for embedding. "
                "Detection results above still reflect the analysis.",
                normal,
            ))

        # ── 4. Chain of Custody ─────────────────────────────────
        story.append(Paragraph("4. CHAIN OF CUSTODY LOG", heading_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#778da9"), spaceAfter=6))

        if custody_logs:
            cust_header = ["Timestamp (UTC)", "Action", "User", "Integrity Hash"]
            cust_data = [cust_header]
            for log in custody_logs[:25]:
                ts = str(log.get("timestamp", ""))[:19]
                action = str(log.get("action", ""))
                user = str(log.get("user_id", ""))[:16]
                h = str(log.get("integrity_hash", log.get("hash", "")))[:16]
                if h:
                    h += "…"
                cust_data.append([ts, action, user, h])
            cust_table = Table(cust_data, colWidths=[1.6*inch, 1.6*inch, 1.3*inch, 2.2*inch])
            cust_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#37474f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#94a3b8")),
                ("PADDING", (0, 0), (-1, -1), 3),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ]))
            story.append(cust_table)
        else:
            story.append(Paragraph(
                "No chain-of-custody entries recorded for this evidence "
                "(MongoDB may be offline or logging was not triggered).",
                normal,
            ))

        # ── 5. Methodology note ─────────────────────────────────
        story.append(Paragraph("5. ANALYSIS METHODOLOGY", heading_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#778da9"), spaceAfter=6))
        story.append(Paragraph(
            "Object detection was performed using a YOLOv8-based deep learning model "
            "(custom forensic weights when available, otherwise official pretrained weights). "
            "Risk scoring combines weapon presence, person detection, confidence levels, "
            "and category counts according to configurable forensic rules. "
            "This automated output is intended as investigative support and must be "
            "reviewed and validated by qualified forensic personnel prior to any legal use.",
            normal,
        ))

        # Footer disclaimer
        story.append(Spacer(1, 14))
        story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#1b263b"), spaceAfter=6))
        story.append(Paragraph(
            "CONFIDENTIAL — For authorized investigative use only. "
            "Generated automatically by AI Crime Scene Evidence Analyzer. "
            "Not a substitute for human forensic examination.",
            footer_style,
        ))
        story.append(Paragraph(
            "AI-generated results are intended to assist investigation and require verification by authorized personnel.",
            normal,
        ))

        doc.build(story)
        return report_path, report_name
