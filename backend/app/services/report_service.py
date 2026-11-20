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

        warnings = risk.get("warnings") or []
        if warnings:
            story.append(Paragraph("<b>Evidence limitations:</b>", normal))
            for warning in warnings:
                story.append(Paragraph(escape(str(warning)), normal))

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
                    f"{float(d['confidence']):.2f}" if d.get("confidence") is not None else "Unavailable",
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
            status = escape(str(classification.get("classification_status", "provisional")))
            story.append(Paragraph(f"Classifier evidence status: <b>{status}</b>. Predictions are provisional and have not been independently validated. A Normal prediction does not prove that no incident occurred.", normal))
            alternatives = classification.get("alternative_predictions") or []
            if alternatives:
                alternative_text = ", ".join(
                    f"{escape(str(item.get('label', 'Unavailable')))} ({float(item['confidence']):.1%})"
                    if item.get("confidence") is not None else escape(str(item.get("label", "Unavailable")))
                    for item in alternatives
                )
                story.append(Paragraph(f"Conflicting temporal-window alternative: {alternative_text}. These are not separate incident findings.", normal))
            info = video_result.get("video_info") or {}
            duration = info.get("duration_sec")
            story.append(Paragraph(
                f"Source video: {escape(os.path.basename(source_filename or 'Not provided'))} · "
                f"Duration: {float(duration):.3f} s" if duration is not None else
                f"Source video: {escape(os.path.basename(source_filename or 'Not provided'))} · Duration unavailable",
                normal,
            ))
            coverage = info.get("analysis_coverage", "unknown")
            coverage_label = {"entire_video": "Entire video", "portion": "Partial video", "unknown": "Unknown"}.get(coverage, "Unknown")
            first_ts = info.get("first_analyzed_timestamp_sec")
            last_ts = info.get("last_analyzed_timestamp_sec")
            interval = info.get("sampling_interval_frames")
            processed = info.get("processed_frames", info.get("frames_analyzed", 0))
            total_frames = info.get("total_frames")
            first_text = f"{float(first_ts):.3f} s" if first_ts is not None else "Unavailable"
            last_text = f"{float(last_ts):.3f} s" if last_ts is not None else "Unavailable"
            fps = info.get("fps")
            fps_text = f"{float(fps):.3f}" if fps is not None else "Unavailable"
            story.append(Paragraph(
                f"VIDEO PROCESSING COVERAGE: <b>{coverage_label}</b> · "
                f"YOLO processed {processed} of {total_frames if total_frames is not None else 'unknown'} frames "
                f"at {fps_text} FPS (sampling interval: {interval if interval is not None else 'unknown'} frames); "
                f"analyzed timestamps: {first_text}–{last_text}.", normal,
            ))
            if info.get("frame_limit_reached") or info.get("duration_limit_reached"):
                story.append(Paragraph("A configured processing limit was reached. YOLO detections represent only the reported interval.", normal))
            classifier_coverage = info.get("classifier_sampling_coverage") or {}
            if classifier_coverage:
                classifier_first = classifier_coverage.get("first_sample_timestamp_sec")
                classifier_last = classifier_coverage.get("last_sample_timestamp_sec")
                classifier_span = (
                    f"{float(classifier_first):.3f}–{float(classifier_last):.3f} s"
                    if classifier_first is not None and classifier_last is not None else
                    f"frames {classifier_coverage.get('first_sample_frame', 'unknown')}–{classifier_coverage.get('last_sample_frame', 'unknown')}"
                )
                classifier_scope = "entire eligible clip" if classifier_coverage.get("covered_entire_video") else "duration-capped clip"
                story.append(Paragraph(
                    f"CNN-LSTM sampling coverage: {classifier_coverage.get('sampled_frame_count', 'unknown')} frames "
                    f"spanning {classifier_span}; {classifier_scope}.", normal,
                ))
            tracking_match = video_result.get("tracking_data_match") or {}
            tracking = video_result.get("tracking") or {}
            story.append(Paragraph(
                f"TRACKING EVIDENCE: <b>{escape(str(tracking_match.get('status') or 'Unavailable'))}</b> · "
                f"{escape(str(tracking_match.get('artifact_type') or tracking.get('method') or 'Artifact unavailable'))} · "
                f"{tracking_match.get('detection_count', tracking.get('detection_count', 0))} detections · "
                f"{tracking_match.get('tracking_id_count', tracking.get('tracking_id_count', 'Unavailable'))} unique tracking IDs. "
                "Tracking provenance is reported separately from classifier confidence.", normal,
            ))
            candidate_events = video_result.get("activity_events") or []
            story.append(Paragraph("TIMESTAMPED ACTIVITY CANDIDATES", heading_style))
            if candidate_events:
                candidate_rows = [["Category", "Start (s)", "End (s)", "Confidence", "Objects", "Track IDs"]]
                for event in candidate_events:
                    candidate_rows.append([
                        escape(str(event.get("category") or "Unavailable")),
                        f"{float(event['start_timestamp_sec']):.3f}" if event.get("start_timestamp_sec") is not None else "Unavailable",
                        f"{float(event['end_timestamp_sec']):.3f}" if event.get("end_timestamp_sec") is not None else "Unavailable",
                        f"{float(event['confidence']):.1%}" if event.get("confidence") is not None else "Unavailable",
                        escape(", ".join(event.get("detected_objects") or []) or "None recorded"),
                        escape(", ".join(map(str, event.get("tracking_ids") or [])) or "None available"),
                    ])
                candidate_table = Table(candidate_rows, colWidths=[0.85*inch, 0.7*inch, 0.7*inch, 0.75*inch, 1.45*inch, 1.2*inch], repeatRows=1)
                candidate_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b263b")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94a3b8")),
                    ("PADDING", (0, 0), (-1, -1), 3),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]))
                story.append(candidate_table)
                story.append(Paragraph("Activity intervals are unverified candidate locations; timestamps do not establish an incident.", small))
            else:
                story.append(Paragraph("No timestamped activity candidate intervals were recorded. This does not establish absence of an incident.", normal))

            evidence_events = video_result.get("evidence_events") or []
            story.append(Paragraph("OBJECT-DETECTION EVENT GROUPS", heading_style))
            story.append(Paragraph(
                "These entries group YOLO object observations across sampled frames. They are unverified object-detection events, not confirmed criminal incidents. Duration is the observed span between first and last sampled detections; a single-frame event has a zero-second observed span.",
                normal,
            ))
            if evidence_events:
                event_rows = [["Observed start–end", "Class", "Span (s)", "Obs.", "Confidence min / mean / max", "Review", "Representative frame"]]
                for event in evidence_events:
                    start = event.get("start_timestamp_sec")
                    end = event.get("end_timestamp_sec")
                    time_span = (
                        f"{float(start):.2f}–{float(end):.2f}"
                        if start is not None and end is not None else
                        f"frames {event.get('first_frame_number', '?')}–{event.get('last_frame_number', '?')}"
                    )
                    summary = event.get("confidence_summary") or {
                        "min": event.get("minimum_confidence"),
                        "mean": event.get("mean_confidence"),
                        "max": event.get("maximum_confidence", event.get("confidence")),
                    }
                    confidence_text = " / ".join(
                        f"{float(summary[key]):.2f}" if summary.get(key) is not None else "—"
                        for key in ("min", "mean", "max")
                    )
                    evidence_path = event.get("representative_evidence_image") or ""
                    evidence_name = str(evidence_path).rsplit("/", 1)[-1] or "Unavailable"
                    evidence_label = f"Frame {event.get('representative_frame_number', '?')} · {evidence_name}"
                    event_rows.append([
                        time_span, escape(str(event.get("evidence_class", event.get("class", "")))),
                        f"{float(event.get('duration_sec') or 0):.2f}",
                        str(event.get("observation_count", 0)), confidence_text,
                        escape(str(event.get("review_status") or "Needs review")), escape(evidence_label),
                    ])
                event_table = Table(event_rows, colWidths=[0.95*inch, 0.65*inch, 0.45*inch, 0.35*inch, 1.2*inch, 0.8*inch, 1.55*inch], repeatRows=1)
                event_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b263b")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94a3b8")),
                    ("PADDING", (0, 0), (-1, -1), 3),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]))
                story.append(event_table)
            else:
                story.append(Paragraph("No object-detection event groups were recorded.", normal))
            limitation = video_result.get("candidate_interval_limitation")
            if limitation:
                story.append(Paragraph("CANDIDATE INTERVAL STATUS", heading_style))
                story.append(Paragraph(escape(str(limitation)), normal))
            timeline = video_result.get("timeline") or video_result.get("event_timeline") or []
            story.append(Paragraph("EVIDENCE TIMELINE", heading_style))
            if timeline:
                timeline_rows = [["Timestamp", "Frame", "Detection", "Confidence", "Track ID", "BBox (x1,y1,x2,y2)"]]
                for item in timeline:
                    seconds = item.get("timestamp_sec")
                    timestamp = f"{float(seconds):.1f} s" if seconds is not None else "Unavailable"
                    confidence_value = item.get("confidence")
                    confidence_text = f"{float(confidence_value):.1%}" if confidence_value is not None else "Unavailable"
                    bbox = item.get("bbox") or []
                    bbox_text = ", ".join(str(value) for value in bbox) or "Unavailable"
                    timeline_rows.append([timestamp, str(item.get("frame_number", "")), escape(str(item.get("class_name", item.get("class", "")))), confidence_text, str(item.get("track_id") if item.get("track_id") is not None else "Unavailable"), escape(bbox_text)])
                timeline_table = Table(timeline_rows, colWidths=[0.75*inch, 0.55*inch, 1.1*inch, 0.85*inch, 0.75*inch, 2.15*inch], repeatRows=1)
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
